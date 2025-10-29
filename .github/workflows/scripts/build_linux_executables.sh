#!/bin/bash

set -euo pipefail

log() { echo "[build] $*"; }

build_executable() {
    local dir="$1"
    local venv_path="$dir/.venv"
    local entry="$2"
    local artifact="$3"
    log "Building $artifact executable..."
    log "Before pushd $dir: $(pwd)"
    pushd "$dir" > /dev/null
    log "Inside $dir directory: $(pwd)"
    uv venv
    source .venv/bin/activate
    uv sync
    uv pip install pyinstaller
    pyinstaller --name "$artifact" --onefile "$entry"
    deactivate
    popd > /dev/null
    log "After popd from $dir: $(pwd)"
}

move_artifact() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
        mv "$src" "$dest"
        log "Created: $dest"
    else
        log "Error: $src not found"
        exit 1
    fi
}

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
COREGATEWAY_VERSION=$(jq -r '.coregateway' .release-please-manifest.json)
CORETELEMETRY_VERSION=$(jq -r '.coretelemetry' .release-please-manifest.json)
COREUI_VERSION=$(jq -r '.coreui' .release-please-manifest.json)

CORERL_ARTIFACT_NAME="linux-corerl"
COREIO_ARTIFACT_NAME="linux-coreio"
COREDINATOR_ARTIFACT_NAME="linux-coredinator"
COREGATEWAY_ARTIFACT_NAME="linux-coregateway"
CORETELEMETRY_ARTIFACT_NAME="linux-coretelemetry"
COREUI_ARTIFACT_NAME="linux-coreui"

if $DEV_BUILD; then
    if [[ -z "$BUILD_NUMBER" ]]; then
        log "Error: --build-number required with --dev"
        exit 1
    fi
    CORERL_ARTIFACT_NAME="${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}-rc.${BUILD_NUMBER}"
    COREIO_ARTIFACT_NAME="${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}-rc.${BUILD_NUMBER}"
    COREDINATOR_ARTIFACT_NAME="${COREDINATOR_ARTIFACT_NAME}-v${COREDINATOR_VERSION}-rc.${BUILD_NUMBER}"
    COREGATEWAY_ARTIFACT_NAME="${COREGATEWAY_ARTIFACT_NAME}-v${COREGATEWAY_VERSION}-rc.${BUILD_NUMBER}"
    CORETELEMETRY_ARTIFACT_NAME="${CORETELEMETRY_ARTIFACT_NAME}-v${CORETELEMETRY_VERSION}-rc.${BUILD_NUMBER}"
    COREUI_ARTIFACT_NAME="${COREUI_ARTIFACT_NAME}-v${COREUI_VERSION}-rc.${BUILD_NUMBER}"
else
    CORERL_ARTIFACT_NAME="${CORERL_ARTIFACT_NAME}-v${CORERL_VERSION}"
    COREIO_ARTIFACT_NAME="${COREIO_ARTIFACT_NAME}-v${COREIO_VERSION}"
    COREDINATOR_ARTIFACT_NAME="${COREDINATOR_ARTIFACT_NAME}-v${COREDINATOR_VERSION}"
    COREGATEWAY_ARTIFACT_NAME="${COREGATEWAY_ARTIFACT_NAME}-v${COREGATEWAY_VERSION}"
    CORETELEMETRY_ARTIFACT_NAME="${CORETELEMETRY_ARTIFACT_NAME}-v${CORETELEMETRY_VERSION}"
    COREUI_ARTIFACT_NAME="${COREUI_ARTIFACT_NAME}-v${COREUI_VERSION}"
fi

log "CoreRL version: $CORERL_VERSION"
log "CoreIO version: $COREIO_VERSION"
log "CoreDinator version: $COREDINATOR_VERSION"
log "CoreGateway version: $COREGATEWAY_VERSION"
log "CoreTelemetry version: $CORETELEMETRY_VERSION"
log "CoreUI version: $COREUI_VERSION"
log "Dev build: $DEV_BUILD"
if $DEV_BUILD; then
    log "Build number: $BUILD_NUMBER"
fi


build_executable "corerl" "corerl/main.py" "$CORERL_ARTIFACT_NAME"
build_executable "coreio" "coreio/main.py" "$COREIO_ARTIFACT_NAME"
build_executable "coredinator" "coredinator/app.py" "$COREDINATOR_ARTIFACT_NAME"
build_executable "coregateway" "coregateway/app.py" "$COREGATEWAY_ARTIFACT_NAME"
build_executable "coretelemetry" "coretelemetry/app.py" "$CORETELEMETRY_ARTIFACT_NAME"

# CoreUI has a special build process 
log "Building CoreUI executable..."
log "Before pushd coreui: $(pwd)"
pushd coreui > /dev/null
log "Inside coreui directory: $(pwd)"
# Build front end first
pushd client > /dev/null
log "Inside coreui/client directory: $(pwd)"
npm install
npm run build
popd > /dev/null
log "After popd from coreui/client: $(pwd)"
# Now build the backend
pushd server > /dev/null
log "Inside coreui/server directory: $(pwd)"
uv venv
source .venv/bin/activate
uv sync
uv pip install pyinstaller
uv run pyinstaller --runtime-tmpdir=. --onefile server/core_ui.py \
    --distpath ../dist \
    --name "$COREUI_ARTIFACT_NAME" \
    --hidden-import=asyncua \
    --hidden-import=fastapi \
    --hidden-import=fastapi.staticfiles \
    --hidden-import=starlette.responses \
    --hidden-import=uvicorn \
    --hidden-import=httpx \
    --hidden-import=server.config_api.config_routes \
    --hidden-import=server.opc_api.opc_routes \
    --hidden-import=server.config_api \
    --hidden-import=server.opc_api \
    --hidden-import=lib_utils.opc.opc_communication \
    --hidden-import=lib_utils.opc \
    --hidden-import=lib_utils \
    --add-data ../client/dist:dist \
    --add-data server:server \
    --add-data ../../libs/lib_utils:lib_utils \
    --paths . \
    --paths server \
    --paths ../../libs \
    --paths ../../libs/lib_utils
deactivate
popd > /dev/null && popd > /dev/null
log "After popd from coreui: $(pwd)"
# End CoreUI build process

mkdir -p dist
move_artifact "./corerl/dist/${CORERL_ARTIFACT_NAME}" "dist/${CORERL_ARTIFACT_NAME}"
move_artifact "./coreio/dist/${COREIO_ARTIFACT_NAME}" "dist/${COREIO_ARTIFACT_NAME}"
move_artifact "./coredinator/dist/${COREDINATOR_ARTIFACT_NAME}" "dist/${COREDINATOR_ARTIFACT_NAME}"
move_artifact "./coregateway/dist/${COREGATEWAY_ARTIFACT_NAME}" "dist/${COREGATEWAY_ARTIFACT_NAME}"
move_artifact "./coretelemetry/dist/${CORETELEMETRY_ARTIFACT_NAME}" "dist/${CORETELEMETRY_ARTIFACT_NAME}"
move_artifact "./coreui/dist/${COREUI_ARTIFACT_NAME}" "dist/${COREUI_ARTIFACT_NAME}"

# Log the artifacts
echo "CORERL_ARTIFACT=${CORERL_ARTIFACT_NAME}" >> dist/artifacts.env
echo "COREIO_ARTIFACT=${COREIO_ARTIFACT_NAME}" >> dist/artifacts.env
echo "COREDINATOR_ARTIFACT=${COREDINATOR_ARTIFACT_NAME}" >> dist/artifacts.env
echo "COREGATEWAY_ARTIFACT=${COREGATEWAY_ARTIFACT_NAME}" >> dist/artifacts.env
echo "CORETELEMETRY_ARTIFACT=${CORETELEMETRY_ARTIFACT_NAME}" >> dist/artifacts.env
echo "COREUI_ARTIFACT=${COREUI_ARTIFACT_NAME}" >> dist/artifacts.env

log "Build completed successfully!"
log "Executables:"
log "  - dist/${CORERL_ARTIFACT_NAME}"
log "  - dist/${COREIO_ARTIFACT_NAME}"
log "  - dist/${COREDINATOR_ARTIFACT_NAME}"
log "  - dist/${COREGATEWAY_ARTIFACT_NAME}"
log "  - dist/${CORETELEMETRY_ARTIFACT_NAME}"
log "  - dist/${COREUI_ARTIFACT_NAME}"