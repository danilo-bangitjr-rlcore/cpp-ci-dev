# CoreUI Client

This directory contains the frontend client for CoreUI, built with React and Vite. It can be developed and run independently from the backend server.

## Development

To start the dev server:

```sh
npm install
npm run dev
```

This will launch the app at [http://localhost:5173/app/](http://localhost:5173/app/)

## Production Build

To generate a production build:

```sh
npm run build
```

The built static files will be output to the `dist/` directory. This `dist` directory is used by the backend server to serve the frontend application.

## Notes

- You can develop and test the UI independently of the backend.
- API requests to `/api` are proxied to `http://localhost:8000` but will require the backend server (fastapi) running 

## Agent Details & Overview Pages

To use the Agents Details and Agents Overview pages in CoreUI, you **must have the Coredinator service running**. These pages rely on Coredinator's REST API to fetch agent status and metadata.

### Quick Start: Running Coredinator

1. **Install dependencies:**
   ```bash
   cd coredinator/
   uv sync
   ```

2. **Start Coredinator (default port 7000):**
   ```bash
   uv run python -m coredinator.app --base-path /path/to/executables
   ```

   - For file logging and rotation:
     ```bash
     uv run python -m coredinator.app --base-path /path/to/executables --log-file /path/to/logs/coredinator.log
     ```

   - For development with auto-reload:
     ```bash
     uv run python -m coredinator.app --base-path /path/to/executables --reload
     ```

   More details and advanced options are available in [coredinator/README.md](../../coredinator/README.md).

### Configuration Requirement

You will need a valid agent configuration YAML file, typically found in the `config/clean/` or `config/raw/mock-configs/` directory. Example configs include:
- `config/clean/mountain_car_continuous.yaml`
- `config/raw/mock-configs/dep_mountain_car_continuous.yaml`

### Current Limitations

- The UI/UX currently only supports the **happy path**: Coredinator must have at least one agent previously started for the agent pages to display correctly.
- **Non-happy paths** (e.g., Coredinator running but no agent started) are **not yet handled** in the UI. You may see empty states, agent/io names not matching, or errors if no agent has been started.

### References

For full details on Coredinator setup, API usage, and service orchestration, see [coredinator/README.md](../../coredinator/README.md).