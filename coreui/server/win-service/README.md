# CoreUI Windows Service

This directory contains scripts and configuration for packaging CoreUI as a Windows service. The service hosts the CoreUI web application, including both the backend (FastAPI) APIs and the frontend static files.

---

## Overview

- **Platform:** Windows only
- **Purpose:** Run CoreUI as a Windows service, serving both backend APIs and the frontend React app.
- **Main Script:** [`windows-service.py`](windows-service.py)  
    Defines a Windows service using `pywin32` that runs the FastAPI server (via Uvicorn), serving both API endpoints and the frontend build output.

---

## Building the Executable

Build a standalone Windows executable using [PyInstaller](https://pyinstaller.org/):

```sh
uv run pyinstaller --runtime-tmpdir=. --onefile windows-service.py --name coreui-service \
    --hidden-import=win32timezone \
    --hidden-import=fastapi.staticfiles \
    --hidden-import=starlette.responses \
    --add-data ../client/dist:dist \
    --add-data ../server/core_ui.py:server
```

Or use the build automation script from the parent directory:

```sh
python build.py build
```

The executable will be created at `./dist/coreui-service.exe`.

---

## Managing the Windows Service

After building, use these commands from this directory:

| Action           | Command                          |
|------------------|----------------------------------|
| **Install**      | `./dist/coreui-service.exe install` |
| **Start**        | `./dist/coreui-service.exe start`   |
| **Stop**         | `./dist/coreui-service.exe stop`    |
| **Remove**       | `./dist/coreui-service.exe remove`  |

---

## Notes

- The service listens on `127.0.0.1:8000` and serves both API endpoints and the frontend app.
- Ensure all dependencies are installed and the frontend is built before packaging.

