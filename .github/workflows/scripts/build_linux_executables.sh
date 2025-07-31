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
CORERL_ARTIFACT_NAME="linux-corerl"
COREIO_ARTIFACT_NAME="linux-coreio"

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
pyinstaller --name "$CORERL_ARTIFACT_NAME" --onefile corerl/main.py
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
pyinstaller --name "$COREIO_ARTIFACT_NAME" --onefile coreio/main.py
deactivate
popd > /dev/null
log "After popd from coreio: $(pwd)"

mkdir -p dist
if [ -f "./corerl/dist/${CORERL_ARTIFACT_NAME}" ]; then
    mv ./corerl/dist/${CORERL_ARTIFACT_NAME} "dist/${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}"
    log "Created: dist/${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}"
else
    log "Error: corerl/dist/${CORERL_ARTIFACT_NAME} not found"
    exit 1
fi

if [ -f "./coreio/dist/${COREIO_ARTIFACT_NAME}" ]; then
    mv ./coreio/dist/${COREIO_ARTIFACT_NAME} "dist/${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}"
    log "Created: dist/${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}"
else
    log "Error: coreio/dist/${COREIO_ARTIFACT_NAME} not found"
    exit 1
fi

log "Build completed successfully!"
log "Executables:"
log "  - dist/${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}"
log "  - dist/${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}"
