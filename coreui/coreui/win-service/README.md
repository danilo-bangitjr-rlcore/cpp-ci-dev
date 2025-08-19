# uv run pyinstaller --runtime-tmpdir=. --onefile windows-service.py --name coreui-service --hidden-import=win32timezone --add-data dist:dist
# ./dist/coreui-service.exe install