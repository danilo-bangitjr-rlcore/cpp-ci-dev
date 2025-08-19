import os
import sys
import uvicorn
from core_ui import CoreUI

# ===== DEBUG INFO =====
print("===== DEBUG INFO =====")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"__file__: {__file__}")
print(f"sys.path: {sys.path}")

# ===== CoreUI app =====
core_ui_instance = CoreUI()
app = core_ui_instance.get_app()  # live app object

# ===== Check frontend folders =====
dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist"))
assets_path = os.path.join(dist_path, "assets")

print(f"Resolved frontend dist folder: {dist_path}")
print(f"Resolved assets folder: {assets_path}")
if not os.path.exists(dist_path):
    print(f"Warning: dist folder does NOT exist at {dist_path}")
if not os.path.exists(assets_path):
    print(f"Warning: assets folder does NOT exist at {assets_path}")

# ===== Start server =====
if __name__ == "__main__":
    print("Starting FastAPI server on 127.0.0.1:8000 ...")
    # Use reload=False because live app object cant be reloaded
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
