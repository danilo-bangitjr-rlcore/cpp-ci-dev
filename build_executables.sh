#!/bin/bash

set -euo pipefail

log() { echo "[build] $*"; }
log "Starting directory: $(pwd)"

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
log "Before pushd corerl: $(pwd)"
pushd corerl > /dev/null
log "Inside corerl directory: $(pwd)"
uv venv
source .venv/bin/activate
uv sync
uv pip install pyinstaller
pyinstaller --name "corerl" --onefile corerl/main.py
deactivate
popd > /dev/null
log "After popd from corerl: $(pwd)"

log "Building CoreIO executable..."
log "Before pushd coreio: $(pwd)"
pushd coreio > /dev/null
log "Inside coreio directory: $(pwd)"
uv venv
source .venv/bin/activate
uv sync
uv pip install pyinstaller
pyinstaller --name "coreio" --onefile coreio/main.py
deactivate
popd > /dev/null
log "After popd from coreio: $(pwd)"

mkdir -p dist
if [ -f "./corerl/dist/corerl" ]; then
    mv ./corerl/dist/corerl "dist/corerl-v${CORERL_VERSION}"
    log "Created: dist/corerl-v${CORERL_VERSION}"
else
    log "Error: corerl/dist/corerl not found"
    exit 1
fi

if [ -f "./coreio/dist/coreio" ]; then
    mv ./coreio/dist/coreio "dist/coreio-v${COREIO_VERSION}"
    log "Created: dist/coreio-v${COREIO_VERSION}"
else
    log "Error: coreio/dist/coreio not found"
    exit 1
fi

log "Build completed successfully!"
log "Executables:"
log "  - dist/corerl-v${CORERL_VERSION}"
log "  - dist/coreio-v${COREIO_VERSION}"
