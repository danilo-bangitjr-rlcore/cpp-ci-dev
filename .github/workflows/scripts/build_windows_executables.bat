@echo off
setlocal enabledelayedexpansion

set LOGPREFIX=[build]
set "LOGCMD=echo %LOGPREFIX%"

%LOGCMD% Starting directory: %CD%

IF NOT EXIST ".release-please-manifest.json" (
    %LOGCMD% Error: .release-please-manifest.json not found
    exit /b 1
)

REM Parse arguments
set DEV_BUILD=false
set BUILD_NUMBER=

:parse_args
if "%~1"=="" goto done_args
if "%~1"=="--dev" (
    set DEV_BUILD=true
    shift
    goto parse_args
)
if "%~1"=="--build-number" (
    set BUILD_NUMBER=%~2
    shift
    shift
    goto parse_args
)
%LOGCMD% Unknown argument: %~1
exit /b 1

:done_args

REM Extract versions from manifest
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).corerl"`) DO set CORERL_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coreio"`) DO set COREIO_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coredinator"`) DO set COREDINATOR_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coregateway"`) DO set COREGATEWAY_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coretelemetry"`) DO set CORETELEMETRY_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coreui"`) DO set COREUI_VERSION=%%A

REM Set base artifact names
set "CORERL_ARTIFACT_NAME=windows-corerl"
set "COREIO_ARTIFACT_NAME=windows-coreio"
set "COREDINATOR_ARTIFACT_NAME=windows-coredinator"
set "COREGATEWAY_ARTIFACT_NAME=windows-coregateway"
set "CORETELEMETRY_ARTIFACT_NAME=windows-coretelemetry"
set "COREUI_ARTIFACT_NAME=windows-coreui"

REM Add version suffixes based on dev build flag
if "%DEV_BUILD%"=="true" (
    if "%BUILD_NUMBER%"=="" (
        %LOGCMD% Error: --build-number required with --dev
        exit /b 1
    )
    set "CORERL_ARTIFACT_NAME=!CORERL_ARTIFACT_NAME!-v%CORERL_VERSION%-rc.%BUILD_NUMBER%"
    set "COREIO_ARTIFACT_NAME=!COREIO_ARTIFACT_NAME!-v%COREIO_VERSION%-rc.%BUILD_NUMBER%"
    set "COREDINATOR_ARTIFACT_NAME=!COREDINATOR_ARTIFACT_NAME!-v%COREDINATOR_VERSION%-rc.%BUILD_NUMBER%"
    set "COREGATEWAY_ARTIFACT_NAME=!COREGATEWAY_ARTIFACT_NAME!-v%COREGATEWAY_VERSION%-rc.%BUILD_NUMBER%"
    set "CORETELEMETRY_ARTIFACT_NAME=!CORETELEMETRY_ARTIFACT_NAME!-v%CORETELEMETRY_VERSION%-rc.%BUILD_NUMBER%"
    set "COREUI_ARTIFACT_NAME=!COREUI_ARTIFACT_NAME!-v%COREUI_VERSION%-rc.%BUILD_NUMBER%"
) else (
    set "CORERL_ARTIFACT_NAME=!CORERL_ARTIFACT_NAME!-v%CORERL_VERSION%"
    set "COREIO_ARTIFACT_NAME=!COREIO_ARTIFACT_NAME!-v%COREIO_VERSION%"
    set "COREDINATOR_ARTIFACT_NAME=!COREDINATOR_ARTIFACT_NAME!-v%COREDINATOR_VERSION%"
    set "COREGATEWAY_ARTIFACT_NAME=!COREGATEWAY_ARTIFACT_NAME!-v%COREGATEWAY_VERSION%"
    set "CORETELEMETRY_ARTIFACT_NAME=!CORETELEMETRY_ARTIFACT_NAME!-v%CORETELEMETRY_VERSION%"
    set "COREUI_ARTIFACT_NAME=!COREUI_ARTIFACT_NAME!-v%COREUI_VERSION%"
)

%LOGCMD% CoreRL version: %CORERL_VERSION%
%LOGCMD% CoreIO version: %COREIO_VERSION%
%LOGCMD% CoreDinator version: %COREDINATOR_VERSION%
%LOGCMD% CoreGateway version: %COREGATEWAY_VERSION%
%LOGCMD% CoreTelemetry version: %CORETELEMETRY_VERSION%
%LOGCMD% CoreUI version: %COREUI_VERSION%
%LOGCMD% Dev build: %DEV_BUILD%
if "%DEV_BUILD%"=="true" (
    %LOGCMD% Build number: %BUILD_NUMBER%
)

REM Build executables
call :build_executable corerl corerl/main.py "!CORERL_ARTIFACT_NAME!"
call :build_executable coreio coreio/main.py "!COREIO_ARTIFACT_NAME!"
call :build_executable coredinator coredinator/app.py "!COREDINATOR_ARTIFACT_NAME!"
call :build_executable coregateway coregateway/app.py "!COREGATEWAY_ARTIFACT_NAME!"
call :build_executable coretelemetry coretelemetry/app.py "!CORETELEMETRY_ARTIFACT_NAME!"

REM CoreUI has a special build process
%LOGCMD% Building CoreUI executable...
%LOGCMD% Before pushd coreui: %CD%
pushd coreui
%LOGCMD% Inside coreui directory: %CD%
REM Build frontend first
pushd client
%LOGCMD% Inside coreui/client directory: %CD%
npm install
npm run build
popd
%LOGCMD% After popd from coreui/client: %CD%
REM Now build the backend
pushd server
%LOGCMD% Inside coreui/server directory: %CD%
uv venv
call .venv\Scripts\activate.bat
uv sync
uv pip install pyinstaller
uv run pyinstaller --runtime-tmpdir=. --onefile server/core_ui.py ^
    --distpath ../dist ^
    --name "!COREUI_ARTIFACT_NAME!" ^
    --hidden-import=fastapi ^
    --hidden-import=fastapi.staticfiles ^
    --hidden-import=starlette.responses ^
    --hidden-import=uvicorn ^
    --hidden-import=httpx ^
    --hidden-import=server.config_api.config_routes ^
    --hidden-import=server.opc_api.opc_routes ^
    --add-data ../client/dist;dist ^
    --add-data server/config_api/mock_configs;mock_configs ^
    --paths server
call deactivate
popd
popd
%LOGCMD% After popd from coreui: %CD%

REM Create dist directory and move artifacts
if not exist dist mkdir dist

call :move_artifact "corerl\dist\!CORERL_ARTIFACT_NAME!.exe" "dist\!CORERL_ARTIFACT_NAME!.exe"
call :move_artifact "coreio\dist\!COREIO_ARTIFACT_NAME!.exe" "dist\!COREIO_ARTIFACT_NAME!.exe"
call :move_artifact "coredinator\dist\!COREDINATOR_ARTIFACT_NAME!.exe" "dist\!COREDINATOR_ARTIFACT_NAME!.exe"
call :move_artifact "coregateway\dist\!COREGATEWAY_ARTIFACT_NAME!.exe" "dist\!COREGATEWAY_ARTIFACT_NAME!.exe"
call :move_artifact "coretelemetry\dist\!CORETELEMETRY_ARTIFACT_NAME!.exe" "dist\!CORETELEMETRY_ARTIFACT_NAME!.exe"
call :move_artifact "coreui\dist\!COREUI_ARTIFACT_NAME!.exe" "dist\!COREUI_ARTIFACT_NAME!.exe"

REM Create artifacts environment file
echo CORERL_ARTIFACT=!CORERL_ARTIFACT_NAME! > dist\artifacts.env
echo COREIO_ARTIFACT=!COREIO_ARTIFACT_NAME! >> dist\artifacts.env
echo COREDINATOR_ARTIFACT=!COREDINATOR_ARTIFACT_NAME! >> dist\artifacts.env
echo COREGATEWAY_ARTIFACT=!COREGATEWAY_ARTIFACT_NAME! >> dist\artifacts.env
echo CORETELEMETRY_ARTIFACT=!CORETELEMETRY_ARTIFACT_NAME! >> dist\artifacts.env
echo COREUI_ARTIFACT=!COREUI_ARTIFACT_NAME! >> dist\artifacts.env

%LOGCMD% Build completed successfully!
%LOGCMD% Executables:
%LOGCMD%   - dist\!CORERL_ARTIFACT_NAME!.exe
%LOGCMD%   - dist\!COREIO_ARTIFACT_NAME!.exe
%LOGCMD%   - dist\!COREDINATOR_ARTIFACT_NAME!.exe
%LOGCMD%   - dist\!COREGATEWAY_ARTIFACT_NAME!.exe
%LOGCMD%   - dist\!CORETELEMETRY_ARTIFACT_NAME!.exe
%LOGCMD%   - dist\!COREUI_ARTIFACT_NAME!.exe

goto :eof

REM Function to build executable
:build_executable
set "dir=%~1"
set "entry=%~2"
set "artifact=%~3"
%LOGCMD% Building %artifact% executable...
%LOGCMD% Before pushd %dir%: %CD%
pushd "%dir%"
%LOGCMD% Inside %dir% directory: %CD%
uv venv
call .venv\Scripts\activate.bat
uv sync
uv pip install pyinstaller
pyinstaller --name "%artifact%" --onefile "%entry%"
call deactivate
popd
%LOGCMD% After popd from %dir%: %CD%
goto :eof

REM Function to move artifact
:move_artifact
set "src=%~1"
set "dest=%~2"
if exist "%src%" (
    move /Y "%src%" "%dest%"
    %LOGCMD% Created: %dest%
) else (
    %LOGCMD% Error: %src% not found
    exit /b 1
)
goto :eof
