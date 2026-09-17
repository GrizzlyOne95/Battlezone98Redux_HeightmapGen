"""Apply the packaged Battlezone tool icon to Tk windows and the Windows taskbar."""
import os
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "GrizzlyOne95.Battlezone98Redux.HeightmapGen"
        )
    except Exception:
        pass

try:
    import tkinter as tk

    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    icon_path = os.path.join(base_path, "branding", "app_icon.png")
    if os.path.exists(icon_path):
        def _wrap_init(cls):
            original = cls.__init__

            def wrapped(self, *args, **kwargs):
                original(self, *args, **kwargs)
                try:
                    image = tk.PhotoImage(file=icon_path)
                    self.iconphoto(True, image)
                    self._battlezone_app_icon = image
                except Exception:
                    pass

            cls.__init__ = wrapped

        _wrap_init(tk.Tk)
        _wrap_init(tk.Toplevel)
except Exception:
    pass
