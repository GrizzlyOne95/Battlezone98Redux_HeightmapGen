from __future__ import annotations

from typing import Optional

from PIL import ImageTk

from .analysis import make_preview, terrain_metrics
from .hg2 import HG2Map
from . import RECIPES, generate
from .settings import GeneratorSettings, random_seed


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("BZR Heightmap Generator")
    root.geometry("1320x900")
    root.configure(bg="#0a0a0a")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    left = ttk.Frame(main, width=330)
    left.pack(side="left", fill="y", padx=(0, 10))
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
    current: dict[str, Optional[HG2Map]] = {"map": None}
    preview_ref = {"image": None}

    ttk.Label(left, text="HEIGHTMAP GENERATOR", font=("Consolas", 13, "bold")).pack(anchor="w", pady=(0, 8))
    ttk.Label(left, text="Hand-authored terrain grammar with direct HG2 export.", wraplength=315).pack(anchor="w", pady=(0, 12))

    ttk.Label(left, text="Terrain Style").pack(anchor="w")
    ttk.Combobox(left, textvariable=vars_["style"], values=list(RECIPES), state="readonly").pack(fill="x", pady=(2, 8))

    dimensions = ttk.Frame(left)
    dimensions.pack(fill="x", pady=2)
    ttk.Label(dimensions, text="Zones X").pack(side="left")
    ttk.Spinbox(dimensions, textvariable=vars_["zones_x"], from_=1, to=8, width=5).pack(side="left", padx=(5, 12))
    ttk.Label(dimensions, text="Zones Z").pack(side="left")
    ttk.Spinbox(dimensions, textvariable=vars_["zones_z"], from_=1, to=8, width=5).pack(side="left", padx=5)

    seed_row = ttk.Frame(left)
    seed_row.pack(fill="x", pady=(5, 10))
    ttk.Label(seed_row, text="Seed").pack(side="left")
    ttk.Entry(seed_row, textvariable=vars_["seed"], width=12).pack(side="left", padx=6)

    def settings_from_ui() -> GeneratorSettings:
        return GeneratorSettings(
            zones_x=max(1, vars_["zones_x"].get()),
            zones_z=max(1, vars_["zones_z"].get()),
            seed=vars_["seed"].get(),
            relief=vars_["relief"].get(),
            vertical_scale=vars_["vertical_scale"].get(),
            naturalization=vars_["naturalization"].get(),
            detail=vars_["detail"].get(),
            plateau_bias=vars_["plateau_bias"].get(),
            feature_density=vars_["feature_density"].get(),
            symmetry=vars_["symmetry"].get(),
            synthetic_pads=max(0, vars_["pads"].get()),
        )

    def randomize_seed() -> None:
        vars_["seed"].set(random_seed())
        do_generate(preserve_seed=True)

    ttk.Button(seed_row, text="Randomize", command=randomize_seed).pack(side="right")
    ttk.Checkbutton(left, text="Fresh random seed each Generate", variable=vars_["fresh_seed"]).pack(anchor="w", pady=(0, 8))

    def slider(label: str, key: str, low: float, high: float, step: float) -> None:
        ttk.Label(left, text=label).pack(anchor="w")
        tk.Scale(
            left, variable=vars_[key], from_=low, to=high, resolution=step,
            orient="horizontal", showvalue=True, bg="#0a0a0a", fg="#d4d4d4",
            highlightthickness=0, troughcolor="#222222",
        ).pack(fill="x")

    slider("Recipe relief", "relief", 0.25, 2.25, 0.05)
    slider("Final terrain contrast / vertical scale", "vertical_scale", 0.35, 1.50, 0.05)
    slider("Naturalization / edge warp", "naturalization", 0.0, 1.0, 0.05)
    slider("Fine detail", "detail", 0.0, 1.0, 0.05)
    slider("Plateau bias", "plateau_bias", 0.0, 1.0, 0.05)
    slider("Feature density", "feature_density", 0.0, 1.0, 0.05)

    ttk.Label(left, text="Synthetic Symmetry").pack(anchor="w", pady=(6, 0))
    ttk.Combobox(
        left, textvariable=vars_["symmetry"],
        values=["None", "Mirror X", "Mirror Z", "2-way rotational", "4-way"],
        state="readonly",
    ).pack(fill="x", pady=(2, 6))

    pad_row = ttk.Frame(left)
    pad_row.pack(fill="x", pady=3)
    ttk.Label(pad_row, text="Objective pads").pack(side="left")
    ttk.Spinbox(pad_row, textvariable=vars_["pads"], from_=0, to=8, width=5).pack(side="right")

    info = ttk.Label(right, text="Generate a terrain to preview it.")
    info.pack(anchor="w", pady=(0, 6))
    canvas = tk.Canvas(right, bg="#050505", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def redraw() -> None:
        terrain = current["map"]
        if terrain is None:
            return
        max_width = max(200, canvas.winfo_width() - 16)
        max_height = max(200, canvas.winfo_height() - 16)
        preview = make_preview(terrain.heights, (max_width, max_height))
        tk_image = ImageTk.PhotoImage(preview)
        preview_ref["image"] = tk_image
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=tk_image, anchor="center")
        metrics = terrain_metrics(terrain.heights)
        world_x, world_z = terrain.world_size
        info.configure(text=(
            f"{terrain.heights.shape[1]}x{terrain.heights.shape[0]} samples | "
            f"{world_x:.0f}x{world_z:.0f} world units | "
            f"height {metrics['min']:.0f}..{metrics['max']:.0f} | "
            f"flat {metrics['exact_flat_pct']:.1f}% | "
            f"median/p95 slope {metrics['median_slope_deg']:.1f}°/{metrics['p95_slope_deg']:.1f}° | "
            f"vertical scale {vars_['vertical_scale'].get():.2f}x"
        ))

    def do_generate(*, preserve_seed: bool = False) -> None:
        try:
            if vars_["fresh_seed"].get() and not preserve_seed:
                vars_["seed"].set(random_seed())
            root.config(cursor="watch")
            root.update_idletasks()
            current["map"] = generate(vars_["style"].get(), settings_from_ui())
            redraw()
        except Exception as exc:
            messagebox.showerror("Generation failed", str(exc))
        finally:
            root.config(cursor="")

    def export_hg2() -> None:
        if current["map"] is None:
            do_generate()
        path = filedialog.asksaveasfilename(defaultextension=".hg2", filetypes=[("Battlezone HG2", "*.hg2")])
        if path and current["map"] is not None:
            current["map"].write(path)

    def export_png() -> None:
        if current["map"] is None:
            do_generate()
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("16-bit PNG", "*.png")])
        if path and current["map"] is not None:
            current["map"].write_png16(path)

    buttons = ttk.Frame(left)
    buttons.pack(fill="x", pady=(12, 0))
    ttk.Button(buttons, text="GENERATE", command=do_generate).pack(fill="x", pady=2)
    ttk.Button(buttons, text="Export HG2...", command=export_hg2).pack(fill="x", pady=2)
    ttk.Button(buttons, text="Export 16-bit PNG...", command=export_png).pack(fill="x", pady=2)
    canvas.bind("<Configure>", lambda _event: redraw())

    do_generate(preserve_seed=True)
    root.mainloop()
