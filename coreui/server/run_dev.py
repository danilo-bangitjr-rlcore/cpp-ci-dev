import os
import sys
from .core_ui import CoreUI

# ===== DEBUG INFO =====
print("===== DEBUG INFO =====")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"__file__: {__file__}")
print(f"sys.path: {sys.path}")

# ===== Check frontend folders =====
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dist_path = os.path.join(base_dir, "client", "dist")
assets_path = os.path.join(dist_path, "assets")

# ===== CoreUI app =====
core_ui_instance = CoreUI()
app = core_ui_instance.get_app()  # live app object

print(f"Resolved frontend dist folder: {dist_path}")
print(f"Resolved assets folder: {assets_path}")
if not os.path.exists(dist_path):
    print(f"Warning: dist folder does NOT exist at {dist_path}")
if not os.path.exists(assets_path):
    print(f"Warning: assets folder does NOT exist at {assets_path}")