# Battlezone98Redux Heightmap Generator v1.0.2

## Summary

This release applies the canonical Heightmap Generator branding throughout the Windows application. The packaged executable, taskbar, Alt-Tab, and Tk window now all use the icon derived from `branding/repo_icon.svg`. There are no terrain-generation changes.

## Highlights

- Rebuild the Windows application icon from the canonical path-based `branding/repo_icon.svg` with a path-capable generator (previous generator assumed the old text/circle artwork).
- Brand the compiled `BZR_Heightmap_Generator.exe` (`--icon bzr_heightmap.ico`).
- Apply the bundled icon to the Tk root window at runtime with frozen (`sys._MEIPASS`) and source-tree resource lookup.
- Set a stable Windows AppUserModelID (`GrizzlyOne95.Battlezone98Redux.HeightmapGen`) so taskbar grouping and icon behavior are reliable.
- Bundle the multi-resolution ICO (16x16 through 256x256) plus PNG in the PyInstaller build with a runtime icon hook.

## Validation

- `python -m unittest discover -s tests` passes.
- Frozen executable `--version` smoke test runs in the release workflow.
