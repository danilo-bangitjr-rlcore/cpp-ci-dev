#!/bin/bash

set -euo pipefail

log() { echo "[build] $*"; }
log "Starting directory: $(pwd)"

if [ ! -f ".release-please-manifest.json" ]; then
    log "Error: .release-please-manifest.json not found"
    exit 1
fi

# Defaults
DEV_BUILD=false
BUILD_NUMBER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_BUILD=true
            shift
            ;;
        --build-number)
            BUILD_NUMBER="$2"
            shift 2
            ;;
        *)
            log "Unknown argument: $1"
            exit 1
            ;;
    esac
done

CORERL_VERSION=$(jq -r '.corerl' .release-please-manifest.json)
COREIO_VERSION=$(jq -r '.coreio' .release-please-manifest.json)
COREDINATOR_VERSION=$(jq -r '.coredinator' .release-please-manifest.json)
CORERL_ARTIFACT_NAME="linux-corerl"
COREIO_ARTIFACT_NAME="linux-coreio"
COREDINATOR_ARTIFACT_NAME="linux-coredinator"

if $DEV_BUILD; then
    if [[ -z "$BUILD_NUMBER" ]]; then
        log "Error: --build-number required with --dev"
        exit 1
    fi
    CORERL_ARTIFACT_NAME="${CORERL_ARTIFACT_NAME}-dev${BUILD_NUMBER}"
    COREIO_ARTIFACT_NAME="${COREIO_ARTIFACT_NAME}-dev${BUILD_NUMBER}"
    COREDINATOR_ARTIFACT_NAME="${COREDINATOR_ARTIFACT_NAME}-dev${BUILD_NUMBER}"
fi

log "CoreRL version: $CORERL_VERSION"
log "CoreIO version: $COREIO_VERSION"
log "CoreDinator version: $COREDINATOR_VERSION"
log "Dev build: $DEV_BUILD"
if $DEV_BUILD; then
    log "Build number: $BUILD_NUMBER"
fi

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

log "Building CoreDinator executable..."
log "Before pushd coredinator: $(pwd)"
pushd coredinator > /dev/null
log "Inside coredinator directory: $(pwd)"
uv venv
source .venv/bin/activate
uv sync
uv pip install pyinstaller
pyinstaller --name "$COREDINATOR_ARTIFACT_NAME" --onefile coredinator/app.py
deactivate
popd > /dev/null
log "After popd from coredinator: $(pwd)"

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

if [ -f "./coredinator/dist/${COREDINATOR_ARTIFACT_NAME}" ]; then
    mv ./coredinator/dist/${COREDINATOR_ARTIFACT_NAME} "dist/${COREDINATOR_ARTIFACT_NAME}-v${COREDINATOR_VERSION}"
    log "Created: dist/${COREDINATOR_ARTIFACT_NAME}-v${COREDINATOR_VERSION}"
else
    log "Error: coredinator/dist/${COREDINATOR_ARTIFACT_NAME} not found"
    exit 1
fi

# Log the artifacts
echo "CORERL_ARTIFACT=${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}" >> dist/artifacts.env
echo "COREIO_ARTIFACT=${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}" >> dist/artifacts.env
echo "COREDINATOR_ARTIFACT=${COREDINATOR_ARTIFACT_NAME}-v${COREDINATOR_VERSION}" >> dist/artifacts.env

log "Build completed successfully!"
log "Executables:"
log "  - dist/${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}"
log "  - dist/${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}"
log "  - dist/${COREDINATOR_ARTIFACT_NAME}-v${COREDINATOR_VERSION}"
