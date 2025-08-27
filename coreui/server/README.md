
# CoreUI Backend Server

This directory contains the FastAPI backend server for CoreUI.

## Development

You can develop and run the backend server independently from the frontend. The backend serves API endpoints and also serves the built frontend static files.

### Running the Backend Server

To start the backend server directly:

```sh
uv run python fastapi dev run_dev.py
```

- **API endpoints:** available under `/api/`
- **Frontend (production build):** served under `/app/`

### Integrated Development

To run both the frontend (Vite dev server) and backend (FastAPI) simultaneously, use the `build.py` script from the `coreui` directory:

- **Vite dev server:** [http://127.0.0.1:5173](http://127.0.0.1:5173)
- **FastAPI backend:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Static File Serving

The backend uses the frontend's `dist` folder to serve the production build. All frontend routes are available under `/app/`.

### API

All backend API endpoints are accessible under `/api/`.

