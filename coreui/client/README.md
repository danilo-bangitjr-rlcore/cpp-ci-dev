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