import subprocess
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
FRONTEND = ROOT / "client"
BACKEND = ROOT / "server"
SERVICE = ROOT / "win-service"
DIST = FRONTEND / "dist"
SERVICE_SCRIPT = SERVICE / "windows-service.py"
CORE_UI_SCRIPT = BACKEND / "core_ui.py"
FASTAPI_DEV_SCRIPT = BACKEND / "run_dev.py"
EXECUTABLE_NAME = "coreui-service"

def run(cmd, cwd=None):
    """Run a shell command"""
    print(f"Running: {cmd}")
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)

def build_frontend():
    print("Building frontend...")
    run("npm install", cwd=FRONTEND)
    run("npm run build", cwd=FRONTEND)

def build_executable():
    print("Building Windows executable with PyInstaller...")
    cmd = (
        f"uv run pyinstaller --runtime-tmpdir=. --onefile {SERVICE_SCRIPT} "
        f"--name {EXECUTABLE_NAME} "
        f"--hidden-import=win32timezone "
        f"--hidden-import=fastapi "
        f"--hidden-import=fastapi.staticfiles "
        f"--hidden-import=starlette.responses "
        f"--add-data {DIST}:dist "
        f"--add-data {CORE_UI_SCRIPT}:server"
    )
    run(cmd, cwd=SERVICE)

def clean():
    print("Cleaning old artifacts...")
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(BACKEND / "__pycache__", ignore_errors=True)
    shutil.rmtree(BACKEND / "build", ignore_errors=True)
    shutil.rmtree(BACKEND / "dist", ignore_errors=True)
    spec_file = BACKEND / f"{EXECUTABLE_NAME}.spec"
    if spec_file.exists():
        spec_file.unlink()

def dev():
    """Run FastAPI (CoreUI) app + Vite in parallel for development"""
    print("Starting development servers... (Ctrl+C to stop)")
    
    build_frontend()
        
    print("Starting development servers... (Ctrl+C to stop)")

    # Start frontend vite
    vite_proc = subprocess.Popen("npm run dev", cwd=FRONTEND, shell=True)

    # Start FastAPI dev
    fastapi_proc = subprocess.Popen(
        f"uv run fastapi dev {FASTAPI_DEV_SCRIPT}", cwd=BACKEND, shell=True
    )

    try:
        vite_proc.wait()
        fastapi_proc.wait()
    except KeyboardInterrupt:
        print("Stopping dev servers...")
        vite_proc.terminate()
        fastapi_proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="Build and dev automation")
    parser.add_argument("command", choices=["build", "clean", "dev"], help="Action to perform")
    args = parser.parse_args()

    if args.command == "clean":
        clean()
    elif args.command == "build":
        clean()
        build_frontend()
        build_executable()
        print("Build complete. Executable is at win-service/dist/coreui-service.exe")
    elif args.command == "dev":
        dev()

if __name__ == "__main__":
    main()
