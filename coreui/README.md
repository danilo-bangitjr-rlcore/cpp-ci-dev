# CoreUI

---

## Architecture Overview

- **Frontend**:  
  - Located in [`client/`](client)  
  - Built with **React** and **Vite**  
  - Production build output is placed in [`client/dist/`](client/dist)

- **Backend**:  
  - Located in [`server/`](server)  
  - Built with **FastAPI**  
  - Serves API endpoints under `/api/`  
  - Serves the frontend static files under `/app/` (from the built `dist` directory)

- **Windows Service**:  
  - Located in [`win-service/`](win-service)  
  - Contains scripts and configuration for packaging the backend and frontend as a single Windows executable using **PyInstaller**

---

## Build & Development Commands

All automation is handled by [`build.py`](build.py).  
Run commands from the `coreui` directory:

### 0. Environment

Activate the uv virtual environment with

On Linux:
```sh
source server/.venv/bin/activate
```

On Windows (`git-bash`):
```sh
source server/.venv/Scripts/activate
```

Install for linux and development work:
```
uv sync
```

If you wish to install `CoreUI` as a windows service, then also install the windows dependencies:
```
uv sync --group win
```

### 1. Clean

Removes all build artifacts and temporary files.

```sh
python build.py clean
```

### 2. Build

Installs frontend dependencies, builds the React app, copies the build output to `dist/`, and packages everything as a Windows executable using pyinstaller

```sh
python build.py build
```

The executable will be at `server/win-service/dist/coreui-service.exe`.

### 3. Development

Builds the frontend, starts the Vite dev server, and runs the FastAPI backend.

```sh
python build.py dev
```
- Vite server runs at [http://127.0.0.1:5173](http://127.0.0.1:5173)
- FastAPI runs at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API endpoints: `/api/`
- Frontend (static): `/app/`

# Developing Modules Independently

Each module—frontend, backend, and Windows service—can also be run and developed independently without using `build.py`.  
Refer to the instructions in the respective directories (`client/`, `server/server`, `server/win-service/`) for standalone development and run commands.
