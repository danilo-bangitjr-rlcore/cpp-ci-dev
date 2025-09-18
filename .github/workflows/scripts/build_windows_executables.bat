@echo off
setlocal enabledelayedexpansion

set LOGPREFIX=[build]
set "LOGCMD=echo %LOGPREFIX%"
set "CORERL_ARTIFACT_NAME=windows-corerl"
set "COREIO_ARTIFACT_NAME=windows-coreio"
set "COREDINATOR_ARTIFACT_NAME=windows-coredinator"

%LOGCMD% Starting directory: %CD%

IF NOT EXIST ".release-please-manifest.json" (
    %LOGCMD% Error: .release-please-manifest.json not found
    exit /b 1
)

FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).corerl"`) DO set CORERL_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coreio"`) DO set COREIO_VERSION=%%A
FOR /F "usebackq tokens=*" %%A IN (`powershell -NoProfile -Command "(Get-Content .release-please-manifest.json | ConvertFrom-Json).coredinator"`) DO set COREDINATOR_VERSION=%%A

%LOGCMD% CoreRL version: %CORERL_VERSION%
%LOGCMD% CoreIO version: %COREIO_VERSION%
%LOGCMD% CoreDinator version: %COREDINATOR_VERSION%

rmdir /s /q dist 2>NUL
rmdir /s /q build 2>NUL
rmdir /s /q corerl\.venv 2>NUL
rmdir /s /q coreio\.venv 2>NUL
rmdir /s /q coredinator\.venv 2>NUL

%LOGCMD% Building CoreRL executable...
pushd corerl
uv venv
call .venv\Scripts\activate.bat
uv sync
uv pip install pyinstaller
pyinstaller --name "%CORERL_ARTIFACT_NAME%" --onefile corerl/main.py
call deactivate
popd

%LOGCMD% Building CoreIO executable...
pushd coreio
uv venv
call .venv\Scripts\activate.bat
uv sync
uv pip install pyinstaller
pyinstaller --name "%COREIO_ARTIFACT_NAME%" --onefile coreio/main.py
call deactivate
popd

%LOGCMD% Building CoreDinator executable...
pushd coredinator
uv venv
call .venv\Scripts\activate.bat
uv sync
uv pip install pyinstaller
pyinstaller --name "%COREDINATOR_ARTIFACT_NAME%" --onefile coredinator/app.py
call deactivate
popd

if not exist dist mkdir dist

if exist corerl\dist\%CORERL_ARTIFACT_NAME%.exe (
    move /Y corerl\dist\%CORERL_ARTIFACT_NAME%.exe dist\%CORERL_ARTIFACT_NAME%-v%CORERL_VERSION%.exe
    %LOGCMD% Created: dist\%CORERL_ARTIFACT_NAME%-v%CORERL_VERSION%.exe
) else (
    %LOGCMD% Error: corerl\dist\%CORERL_ARTIFACT_NAME%.exe not found
    exit /b 1
)

if exist coreio\dist\%COREIO_ARTIFACT_NAME%.exe (
    move /Y coreio\dist\%COREIO_ARTIFACT_NAME%.exe dist\%COREIO_ARTIFACT_NAME%-v%COREIO_VERSION%.exe
    %LOGCMD% Created: dist\%COREIO_ARTIFACT_NAME%-v%COREIO_VERSION%.exe
) else (
    %LOGCMD% Error: coreio\dist\%COREIO_ARTIFACT_NAME%.exe not found
    exit /b 1
)

if exist coredinator\dist\%COREDINATOR_ARTIFACT_NAME%.exe (
    move /Y coredinator\dist\%COREDINATOR_ARTIFACT_NAME%.exe dist\%COREDINATOR_ARTIFACT_NAME%-v%COREDINATOR_VERSION%.exe
    %LOGCMD% Created: dist\%COREDINATOR_ARTIFACT_NAME%-v%COREDINATOR_VERSION%.exe
) else (
    %LOGCMD% Error: coredinator\dist\%COREDINATOR_ARTIFACT_NAME%.exe not found
    exit /b 1
)

%LOGCMD% Build completed successfully!
%LOGCMD% Executables:
%LOGCMD%   - dist\%CORERL_ARTIFACT_NAME%-v%CORERL_VERSION%.exe
%LOGCMD%   - dist\%COREIO_ARTIFACT_NAME%-v%COREIO_VERSION%.exe
%LOGCMD%   - dist\%COREDINATOR_ARTIFACT_NAME%-v%COREDINATOR_VERSION%.exe