# Web Client

[Wiki](https://github.com/rlcoretech/core-rl/wiki#developing-on-corerl-web-gui)

This is the web client for CoreRL. We aim to provide a graphical user interface for setting up and interacting with the
running CoreRL agent/experiment instance.

## Auto Generating API Client Types

Our application uses [`OpenAPI TS`](https://openapi-ts.dev/introduction) to generate our client TypeScript types from our FastAPI server.

With our web server running, the schemas can be regenerated with:

```bash
npm run build:api
openapi-typescript http://localhost:8000/openapi.json -o ./src/api-schema.d.ts
```
