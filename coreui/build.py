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
DEFAULT_CONFIG_PATH = ROOT / ".." / "config"

DEV_GRACE_SECONDS = 5.0

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
    
    configs_dir = BACKEND / "server" / "config_api" / "mock_configs"
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
        f"--hidden-import=starlette.responses "
        f"--hidden-import=scipy._cyutility "
        f"--add-data {DIST}:dist "
        f"--add-data {configs_dir}:mock_configs "
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

def _start_service(name: str, cmd: str, cwd: str | Path):
    """Start a service in its directory and return its process"""
    print(f"Starting {name}...")
    return subprocess.Popen(cmd, cwd=cwd, shell=True)

def _block_on_processes(procs: list[subprocess.Popen[bytes]]):
    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=DEV_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()

def dev():
    """Run FastAPI (CoreUI) app + Vite in parallel for development"""
    print("Starting development servers... (Ctrl+C to stop)")
    build_frontend()

    procs = [
        _start_service("Vite", "npm run dev", FRONTEND),
        _start_service("FastAPI", f"uv run fastapi dev {FASTAPI_DEV_SCRIPT}", BACKEND),
    ]

    _block_on_processes(procs)

def dev_stack(coredinator_path: Path, coretelemetry_path: Path):
    print("Starting development servers... (Ctrl+C to stop)")
    build_frontend()

    # Start microservices
    procs = [
        _start_service("Vite", "npm run dev", FRONTEND),
        _start_service("FastAPI", f"uv run fastapi dev {FASTAPI_DEV_SCRIPT}", BACKEND),
        _start_service("CoreGateway", "uv run python coregateway/app.py", "../coregateway"),
        _start_service("CoreDinator", f"uv run python coredinator/app.py --base-path {coredinator_path}", "../coredinator"),
        _start_service("CoreTelemetry", f"uv run python coretelemetry/app.py --config-path {coretelemetry_path}", "../coretelemetry"),
    ]

    _block_on_processes(procs)

def main():
    parser = argparse.ArgumentParser(description="Build and dev automation")
    parser.add_argument("command", choices=["build", "clean", "dev", "dev-stack"], help="Action to perform")
    parser.add_argument("--coredinator-path", default=DEFAULT_CONFIG_DIR, help="--base-path for Coredinator")
    parser.add_argument("--coretelemetry-path", default=DEFAULT_CONFIG_DIR, help="--config-path for CoreTelemetry")

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
    elif args.command == "dev-stack":
        dev_stack(Path(args.coredinator_path), Path(args.coretelemetry_path))

if __name__ == "__main__":
    main()
