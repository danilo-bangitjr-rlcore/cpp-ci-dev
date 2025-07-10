#!/bin/bash

set -euo pipefail

log() { echo "[build] $*"; }

start_time=$(date +%s)

echo "Install uv"
curl -fsSL https://get.uv.dev | sh

if [ ! -f ".release-please-manifest.json" ]; then
    log "Error: .release-please-manifest.json not found"
    exit 1
fi

CORERL_VERSION=$(jq -r '.corerl' .release-please-manifest.json)
COREIO_VERSION=$(jq -r '.coreio' .release-please-manifest.json)

log "CoreRL version: $CORERL_VERSION"
log "CoreIO version: $COREIO_VERSION"

rm -rf dist build corerl/.venv coreio/.venv

log "Building CoreRL executable..."
pushd corerl > /dev/null
uv venv
source .venv/bin/activate
uv sync
uv pip install pyinstaller
pyinstaller --name "corerl" --onefile corerl/main.py
deactivate
popd > /dev/null

log "Building CoreIO executable..."
pushd coreio > /dev/null
uv venv
source .venv/bin/activate
uv sync
uv pip install pyinstaller
pyinstaller --name "coreio" --onefile coreio/main.py
deactivate
popd > /dev/null

if [ -f "corerl/dist/corerl" ]; then
    mv corerl/dist/corerl "dist/corerl-v${CORERL_VERSION}"
    log "Created: dist/corerl-v${CORERL_VERSION}"
else
    log "Error: corerl/dist/corerl not found"
    exit 1
fi

if [ -f "coreio/dist/coreio" ]; then
    mv coreio/dist/coreio "dist/coreio-v${COREIO_VERSION}"
    log "Created: dist/coreio-v${COREIO_VERSION}"
else
    log "Error: coreio/dist/coreio not found"
    exit 1
fi

log "Build completed successfully!"
log "Executables:"
log "  - dist/corerl-v${CORERL_VERSION}"
log "  - dist/coreio-v${COREIO_VERSION}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
log "Total build time: ${elapsed}s"