from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from PIL import Image, ImageTk

from .analysis import terrain_metrics
from .gui_logic import (
    DEBOUNCE_MS,
    LatestJobCoordinator,
    RawCacheKey,
    apply_cached_contrast,
    raw_generation_settings,
)
from .hg2 import HG2Map
from .lgt import compute_lgt_lightmap, write_lgt
from .preview import make_hg2_height_image, make_lgt_preview_image, make_shaded_preview_fullres
from . import RECIPES, generate
from .settings import GeneratorSettings, random_seed


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("BZR Heightmap Generator — Live HG2 / LGT Preview")
    root.geometry("1380x920")
    root.configure(bg="#0a0a0a")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    main = ttk.Frame(root, padding=8)
    main.pack(fill="both", expand=True)
    left = ttk.Frame(main, width=360)
    left.pack(side="left", fill="y", padx=(0, 8))
    left.pack_propagate(False)
    right = ttk.Frame(main)
    right.pack(side="right", fill="both", expand=True)

    vars_ = {
        "style": tk.StringVar(value="Terraced Labyrinth"),
        "zones_x": tk.IntVar(value=3),
        "zones_z": tk.IntVar(value=3),
        "seed": tk.IntVar(value=random_seed()),
        "fresh_seed": tk.BooleanVar(value=True),
        "relief": tk.DoubleVar(value=1.0),
        "vertical_scale": tk.DoubleVar(value=1.0),
        "naturalization": tk.DoubleVar(value=0.65),
        "detail": tk.DoubleVar(value=0.55),
        "plateau_bias": tk.DoubleVar(value=0.5),
        "feature_density": tk.DoubleVar(value=0.5),
        "symmetry": tk.StringVar(value="None"),
        "pads": tk.IntVar(value=0),
    }

    # Caching: raw_heights before vertical contrast, and full-res preview images.
    current: dict[str, Optional[object]] = {"map": None, "raw": None, "raw_key": None}
    # Full-res PIL images before thumbnailing (for resize without recompute)
    preview_full = {"hg2": None, "lgt": None, "shaded": None}
    preview_ref = {"hg2": None, "lgt": None, "shaded": None}
    debounce = {"after_id": None}
    poll = {"after_id": None}
    coordinator = LatestJobCoordinator()
    worker_results: queue.Queue[tuple] = queue.Queue()

    # --- Left panel: Header ---
    ttk.Label(left, text="HEIGHTMAP GENERATOR", font=("Consolas", 13, "bold")).pack(anchor="w", pady=(0, 4))
    ttk.Label(left, text="Live HG2 Height / LGT Lighting / Shaded previews.", wraplength=345).pack(anchor="w", pady=(0, 6))
    ttk.Label(left, text="Hand-authored terrain grammar with direct HG2 export.", wraplength=345, font=("Consolas", 8), foreground="#888888").pack(anchor="w", pady=(0, 8))

    # --- Basic controls (prominent) ---
    basic = ttk.LabelFrame(left, text=" Basic ", padding=8)
    basic.pack(fill="x", pady=(0, 6))

    ttk.Label(basic, text="Terrain Style").pack(anchor="w")
    cb_style = ttk.Combobox(basic, textvariable=vars_["style"], values=list(RECIPES), state="readonly")
    cb_style.pack(fill="x", pady=(2, 6))

    dims = ttk.Frame(basic)
    dims.pack(fill="x", pady=2)
    ttk.Label(dims, text="Zones X").pack(side="left")
    sp_x = ttk.Spinbox(dims, textvariable=vars_["zones_x"], from_=1, to=8, width=5)
    sp_x.pack(side="left", padx=(5, 12))
    ttk.Label(dims, text="Zones Z").pack(side="left")
    sp_z = ttk.Spinbox(dims, textvariable=vars_["zones_z"], from_=1, to=8, width=5)
    sp_z.pack(side="left", padx=5)

    seed_row = ttk.Frame(basic)
    seed_row.pack(fill="x", pady=(6, 4))
    ttk.Label(seed_row, text="Seed").pack(side="left")
    ent_seed = ttk.Entry(seed_row, textvariable=vars_["seed"], width=12)
    ent_seed.pack(side="left", padx=6)
    # Seed display remains visible at all times via vars_["seed"]
    def randomize_seed() -> None:
        vars_["seed"].set(random_seed())

    ttk.Button(seed_row, text="Randomize", command=randomize_seed).pack(side="right")

    ttk.Checkbutton(basic, text="Fresh random seed each Generate", variable=vars_["fresh_seed"]).pack(anchor="w", pady=(2, 4))

    # Terrain Contrast / Vertical Relief — prominent with help
    ttk.Label(basic, text="Terrain Contrast / Vertical Relief", font=("Consolas", 9, "bold")).pack(anchor="w", pady=(6, 0))
    ttk.Label(
        basic,
        text="Lower = compressed height differences, shallower grades in-game. 1.0 = default. Higher = exaggerated relief. Exact flats stay exact.",
        wraplength=335,
        font=("Consolas", 7),
        foreground="#aaaaaa",
    ).pack(anchor="w")
    scale_contrast = tk.Scale(
        basic,
        variable=vars_["vertical_scale"],
        from_=0.35,
        to=1.50,
        resolution=0.05,
        orient="horizontal",
        showvalue=True,
        bg="#0a0a0a",
        fg="#d4d4d4",
        highlightthickness=0,
        troughcolor="#222222",
        command=lambda _v: schedule_generate(preserve_seed=True),
    )
    scale_contrast.pack(fill="x")

    # --- Advanced (collapsible via button) ---
    adv_visible = {"on": False}
    adv_frame = ttk.Frame(left)

    def toggle_advanced() -> None:
        adv_visible["on"] = not adv_visible["on"]
        if adv_visible["on"]:
            adv_frame.pack(fill="x", pady=(4, 0))
            btn_adv.configure(text="Advanced ▾")
        else:
            adv_frame.pack_forget()
            btn_adv.configure(text="Advanced ▸")

    btn_adv = ttk.Button(left, text="Advanced ▸", command=toggle_advanced)
    btn_adv.pack(fill="x", pady=(4, 0))

    def slider(parent, label: str, key: str, low: float, high: float, step: float) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 0))
        tk.Scale(
            parent,
            variable=vars_[key],
            from_=low,
            to=high,
            resolution=step,
            orient="horizontal",
            showvalue=True,
            bg="#0a0a0a",
            fg="#d4d4d4",
            highlightthickness=0,
            troughcolor="#222222",
            command=lambda _v: schedule_generate(preserve_seed=True),
        ).pack(fill="x")

    slider(adv_frame, "Recipe relief", "relief", 0.25, 2.25, 0.05)
    slider(adv_frame, "Naturalization / edge warp", "naturalization", 0.0, 1.0, 0.05)
    slider(adv_frame, "Fine detail", "detail", 0.0, 1.0, 0.05)
    slider(adv_frame, "Plateau bias", "plateau_bias", 0.0, 1.0, 0.05)
    slider(adv_frame, "Feature density", "feature_density", 0.0, 1.0, 0.05)

    ttk.Label(adv_frame, text="Synthetic Symmetry").pack(anchor="w", pady=(6, 0))
    cb_sym = ttk.Combobox(
        adv_frame,
        textvariable=vars_["symmetry"],
        values=["None", "Mirror X", "Mirror Z", "2-way rotational", "4-way"],
        state="readonly",
    )
    cb_sym.pack(fill="x", pady=(2, 6))
    cb_sym.bind("<<ComboboxSelected>>", lambda _e: schedule_generate(preserve_seed=True))

    pad_row = ttk.Frame(adv_frame)
    pad_row.pack(fill="x", pady=3)
    ttk.Label(pad_row, text="Objective pads").pack(side="left")
    sp_pads = ttk.Spinbox(pad_row, textvariable=vars_["pads"], from_=0, to=8, width=5, command=lambda: schedule_generate(preserve_seed=True))
    sp_pads.pack(side="right")
    # Spinbox text changes don't fire command reliably; bind var trace for pads
    vars_["pads"].trace_add("write", lambda *_: schedule_generate(preserve_seed=True))

    # --- Generation / Export ---
    btns = ttk.Frame(left)
    btns.pack(fill="x", pady=(10, 0))
    # info line for seed visibility
    seed_info = ttk.Label(left, text="", font=("Consolas", 8), foreground="#9cdcfe")
    seed_info.pack(anchor="w", pady=(6, 0))

    # Bind style combobox and zones spinboxes to live preview
    cb_style.bind("<<ComboboxSelected>>", lambda _e: schedule_generate(preserve_seed=True))
    # Trace zones vars also live (but clamp)
    for k in ("zones_x", "zones_z"):
        vars_[k].trace_add("write", lambda *_: schedule_generate(preserve_seed=True))
    vars_["seed"].trace_add("write", lambda *_: schedule_generate(preserve_seed=True))

    # --- Right: Preview area with tabs ---
    info = ttk.Label(right, text="Generate a terrain to preview it.", font=("Consolas", 8))
    info.pack(anchor="w", pady=(0, 4))

    notebook = ttk.Notebook(right)
    notebook.pack(fill="both", expand=True)

    tab_hg2 = ttk.Frame(notebook)
    tab_lgt = ttk.Frame(notebook)
    tab_shaded = ttk.Frame(notebook)
    notebook.add(tab_hg2, text="  HG2 Height  ")
    notebook.add(tab_lgt, text="  LGT Lighting  ")
    notebook.add(tab_shaded, text="  Shaded  ")

    canvases = {}
    for tab, name in [(tab_hg2, "hg2"), (tab_lgt, "lgt"), (tab_shaded, "shaded")]:
        c = tk.Canvas(tab, bg="#050505", highlightthickness=0)
        c.pack(fill="both", expand=True)
        canvases[name] = c

    # Tooltips / help for contrast already provided above; additional labels
    ttk.Label(tab_hg2, text="Raw HG2 height field (0..4095 fixed mapping, no percentile renormalization).", font=("Consolas", 7), foreground="#888888").pack(side="bottom", fill="x")
    ttk.Label(tab_lgt, text="BZ LGT-style lighting (slope normals + NW sun + 25% ambient floor). Not an engine-valid .LGT export unless verified.", font=("Consolas", 7), foreground="#888888").pack(side="bottom", fill="x")
    ttk.Label(tab_shaded, text="Combined elevation + hillshade (legacy preview).", font=("Consolas", 7), foreground="#888888").pack(side="bottom", fill="x")

    def settings_from_ui() -> GeneratorSettings:
        return GeneratorSettings(
            zones_x=max(1, int(vars_["zones_x"].get())),
            zones_z=max(1, int(vars_["zones_z"].get())),
            seed=int(vars_["seed"].get()),
            relief=float(vars_["relief"].get()),
            vertical_scale=float(vars_["vertical_scale"].get()),
            naturalization=float(vars_["naturalization"].get()),
            detail=float(vars_["detail"].get()),
            plateau_bias=float(vars_["plateau_bias"].get()),
            feature_density=float(vars_["feature_density"].get()),
            symmetry=str(vars_["symmetry"].get()),
            synthetic_pads=max(0, int(vars_["pads"].get())),
        )

    def update_seed_info() -> None:
        # Keep exact resolved seed visible at all times
        try:
            seed_info.configure(text=f"Seed: {int(vars_['seed'].get())}   |   Style: {vars_['style'].get()}   |   {vars_['zones_x'].get()}×{vars_['zones_z'].get()} zones")
        except Exception:
            seed_info.configure(text=f"Style: {vars_['style'].get()}")

    def build_preview_images(terrain: HG2Map) -> dict[str, Image.Image]:
        """Build PIL previews off-thread; Tk images are created in redraw()."""
        return {
            "hg2": make_hg2_height_image(terrain.heights),
            "lgt": make_lgt_preview_image(terrain.heights, terrain.zones_x, terrain.zones_z, lgt_zone_size=128),
            "shaded": make_shaded_preview_fullres(terrain.heights),
        }

    def redraw() -> None:
        # Resizing the window does not regenerate terrain — just re-thumbnails cached images.
        terrain = current.get("map")  # type: ignore
        if terrain is None or preview_full["hg2"] is None:
            return
        for name, canvas in canvases.items():
            max_w = max(120, canvas.winfo_width() - 8)
            max_h = max(120, canvas.winfo_height() - 8)
            full = preview_full[name]
            if full is None:
                continue
            # Thumbnail preserving aspect; use LANCZOS for shaded, NEAREST for height/lgt to keep exactness?
            thumb = full.copy()
            thumb.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(thumb)
            preview_ref[name] = tk_img
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=tk_img, anchor="center")
        # Update info line
        try:
            t = terrain  # type: HG2Map
            metrics = terrain_metrics(t.heights)  # type: ignore
            world_x, world_z = t.world_size  # type: ignore
            info.configure(
                text=(
                    f"{t.heights.shape[1]}×{t.heights.shape[0]} samples | "
                    f"{world_x:.0f}×{world_z:.0f} world units | "
                    f"height {metrics['min']:.0f}..{metrics['max']:.0f} | "
                    f"flat {metrics['exact_flat_pct']:.1f}% | "
                    f"median/p95 slope {metrics['median_slope_deg']:.1f}°/{metrics['p95_slope_deg']:.1f}° | "
                    f"vertical scale {vars_['vertical_scale'].get():.2f}×"
                )
            )
        except Exception:
            pass
        update_seed_info()

    def generate_in_worker(
        style_name: str,
        settings: GeneratorSettings,
        revision: int,
        cached_raw: HG2Map | None,
        cached_key: RawCacheKey | None,
    ) -> None:
        """Worker entry point. It never reads Tk state or calls Tk methods."""
        started = time.perf_counter()
        try:
            raw_key = RawCacheKey.from_settings(style_name, settings)
            reused_raw = cached_raw is not None and cached_key == raw_key
            raw = cached_raw if reused_raw else generate(style_name, raw_generation_settings(settings))
            if raw is None:
                raise RuntimeError("terrain generation did not produce a map")
            terrain = apply_cached_contrast(raw, settings.vertical_scale)
            images = build_preview_images(terrain)
            worker_results.put((revision, raw_key, raw, terrain, images, reused_raw, time.perf_counter() - started, None))
        except Exception as exc:
            worker_results.put((revision, None, None, None, None, False, time.perf_counter() - started, exc))

    def do_generate() -> None:
        """Start the newest pending request if the single worker is idle."""
        revision = coordinator.start_latest()
        if revision is None:
            return
        try:
            settings = settings_from_ui()
            style_name = vars_["style"].get()
        except Exception as exc:
            coordinator.finish(revision)
            messagebox.showerror("Settings error", str(exc))
            return
        root.config(cursor="watch")
        threading.Thread(
            target=generate_in_worker,
            args=(style_name, settings, revision, current.get("raw"), current.get("raw_key")),
            daemon=True,
        ).start()
        update_seed_info()

    def schedule_generate(*, preserve_seed: bool = True, immediate: bool = False) -> None:
        """Debounce controls while retaining exactly one newest pending job."""
        if coordinator.closing:
            return
        if vars_["fresh_seed"].get() and not preserve_seed:
            vars_["seed"].set(random_seed())
        coordinator.schedule()
        if debounce["after_id"] is not None:
            try:
                root.after_cancel(debounce["after_id"])
            except Exception:
                pass
            debounce["after_id"] = None
        if immediate:
            do_generate()
        else:
            def fire() -> None:
                debounce["after_id"] = None
                do_generate()

            debounce["after_id"] = root.after(DEBOUNCE_MS, fire)

    def poll_worker_results() -> None:
        """Main-thread bridge: accept only latest results and launch queued work."""
        while True:
            try:
                revision, raw_key, raw, terrain, images, reused_raw, elapsed, error = worker_results.get_nowait()
            except queue.Empty:
                break

            needs_followup = coordinator.finish(revision)
            if coordinator.closing:
                return

            # A stale full generation can still seed the contrast cache when
            # the newest UI state has the same generation-affecting key.
            if raw is not None and raw_key is not None:
                try:
                    latest_key = RawCacheKey.from_settings(vars_["style"].get(), settings_from_ui())
                except Exception:
                    latest_key = None
                if raw_key == latest_key:
                    current["raw"] = raw
                    current["raw_key"] = raw_key

            if coordinator.accepts(revision):
                if error is not None:
                    messagebox.showerror("Generation failed", str(error))
                else:
                    current["map"] = terrain
                    preview_full.update(images)
                    redraw()
                    mode = "contrast cache" if reused_raw else "full generation"
                    info.configure(text=f"{info.cget('text')} | {mode} {elapsed * 1000.0:.0f} ms")

            if needs_followup and coordinator.active_revision is None and debounce["after_id"] is None:
                root.after_idle(do_generate)

        if coordinator.active_revision is None and not coordinator.pending:
            root.config(cursor="")
        poll["after_id"] = root.after(25, poll_worker_results)

    # --- Exports ---
    def export_hg2() -> None:
        if current.get("map") is None:
            schedule_generate(preserve_seed=True, immediate=True)
            return
        path = filedialog.asksaveasfilename(defaultextension=".hg2", filetypes=[("Battlezone HG2", "*.hg2"), ("All files", "*.*")])
        if path and isinstance(current.get("map"), HG2Map):
            try:
                current["map"].write(path)  # type: ignore
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def export_png16() -> None:
        if current.get("map") is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("16-bit PNG", "*.png"), ("All files", "*.*")])
        if path and isinstance(current.get("map"), HG2Map):
            try:
                current["map"].write_png16(path)  # type: ignore
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def export_hg2_display_png() -> None:
        m = current.get("map")
        if not isinstance(m, HG2Map):
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                make_hg2_height_image(m.heights).save(path)
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def export_lgt_preview_png() -> None:
        m = current.get("map")
        if not isinstance(m, HG2Map):
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                make_lgt_preview_image(m.heights, m.zones_x, m.zones_z).save(path)
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def export_shaded_preview_png() -> None:
        m = current.get("map")
        if not isinstance(m, HG2Map):
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                make_shaded_preview_fullres(m.heights).save(path)
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def export_lgt() -> None:
        m = current.get("map")
        if not isinstance(m, HG2Map):
            return
        path = filedialog.asksaveasfilename(defaultextension=".lgt", filetypes=[("Battlezone LGT", "*.lgt"), ("All files", "*.*")])
        if path:
            try:
                # Export at the 256-per-zone resolution used by most observed
                # Redux LGT pairs. The 128-per-zone legacy path remains in the API.
                # This export is provided for completeness; correctness is based on local
                # HG2/LGT pair validation (see docs/HG2_CORPUS_ANALYSIS).  Not claimed as
                # engine-verified without further game testing.
                light = compute_lgt_lightmap(m.heights, m.zones_x, m.zones_z, lgt_zone_size=256)
                write_lgt(path, light, m.zones_x, m.zones_z)
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    # Buttons
    ttk.Button(btns, text="GENERATE  (New Terrain)", command=lambda: schedule_generate(preserve_seed=False, immediate=True)).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export HG2...", command=export_hg2).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export 16-bit Height PNG...", command=export_png16).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export HG2 Height PNG...", command=export_hg2_display_png).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export LGT Preview PNG...", command=export_lgt_preview_png).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export Shaded Preview PNG...", command=export_shaded_preview_png).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export LGT 256/zone (experimental)...", command=export_lgt).pack(fill="x", pady=2)

    # Help text for exports
    ttk.Label(left, text="HG2 stores heights. LGT preview shows terrain lighting, not height. Lowering vertical relief makes slopes less severe without blurring.", wraplength=345, font=("Consolas", 7), foreground="#777777").pack(anchor="w", pady=(8, 0))

    # Canvas resize should not regenerate terrain — just re-thumbnail
    for c in canvases.values():
        c.bind("<Configure>", lambda _e: redraw())

    def on_close() -> None:
        coordinator.close()
        if debounce["after_id"] is not None:
            try:
                root.after_cancel(debounce["after_id"])
            except Exception:
                pass
        if poll["after_id"] is not None:
            try:
                root.after_cancel(poll["after_id"])
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Polling is created on the Tk thread; workers communicate only by queue.
    poll["after_id"] = root.after(25, poll_worker_results)
    schedule_generate(preserve_seed=True, immediate=True)
    root.mainloop()
