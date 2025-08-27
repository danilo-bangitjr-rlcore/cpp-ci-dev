import os
import subprocess
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
FRONTEND = ROOT / "client"
BACKEND = ROOT / "server"
SERVICE = ROOT / BACKEND / "win-service"
DIST = FRONTEND / "dist"
SERVICE_SCRIPT = SERVICE / "windows-service.py"
FASTAPI_DEV_SCRIPT = BACKEND / "run_dev.py"
EXECUTABLE_NAME = "coreui-service"
SERVER_LIB = BACKEND / "server" / ".."

def run(cmd: str, cwd: Path | None=None):
    """Run a shell command"""
    print(f"Running: {cmd}")
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)

def build_frontend():
    print("Building frontend...")
    run("npm install", cwd=FRONTEND)
    run("npm run build", cwd=FRONTEND)

def build_executable():
    print("Building Windows executable with PyInstaller...")
    
    venv_path = BACKEND / ".venv"
    venv_active = "VIRTUAL_ENV" in os.environ and Path(os.environ["VIRTUAL_ENV"]).resolve() == venv_path.resolve()
    if not venv_active:
        print("Please activate the .venv before running this command.")
        exit(1)
    
    cmd = (
        f"uv run pyinstaller --runtime-tmpdir=. --onefile {SERVICE_SCRIPT} "
        f"--name {EXECUTABLE_NAME} "
        f"--hidden-import=win32timezone "
        f"--hidden-import=fastapi "
        f"--hidden-import=fastapi.staticfiles "
        f"--add-data {DIST}:dist "
        f"--paths {SERVER_LIB}"
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
