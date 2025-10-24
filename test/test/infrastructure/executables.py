import os
from pathlib import Path

import pytest
from PyInstaller.__main__ import run as pyinstaller_run


def _get_monorepo_root() -> Path:
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".release-please-manifest.json").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate monorepo root (no .release-please-manifest.json found)")


def _get_bin_dir() -> Path:
    monorepo_root = _get_monorepo_root()
    bin_dir = monorepo_root / "test" / "bin"
    bin_dir.mkdir(exist_ok=True)
    return bin_dir


def _build_service_executable(
    service_name: str,
    entry_point: str,
    hidden_imports: list[str] | None = None,
    collect_all: list[str] | None = None,
) -> Path:
    monorepo_root = _get_monorepo_root()
    service_dir = monorepo_root / service_name

    if not service_dir.exists():
        raise FileNotFoundError(f"Service directory not found: {service_dir}")

    bin_dir = _get_bin_dir()

    platform = "windows" if os.name == "nt" else "linux"
    executable_base = f"{platform}-{service_name}-v0.1.0"
    executable_name = f"{executable_base}.exe" if os.name == "nt" else executable_base
    executable_path = bin_dir / executable_name

    if executable_path.exists():
        return executable_path

    entry_file = service_dir / entry_point
    if not entry_file.exists():
        raise FileNotFoundError(f"Entry point not found: {entry_file}")

    build_dir = bin_dir / "build" / service_name
    spec_dir = bin_dir / "spec"

    pyinstaller_args = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        executable_base,
        "--distpath",
        str(bin_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(spec_dir),
        "--paths",
        str(service_dir),
    ]

    if hidden_imports:
        for imp in hidden_imports:
            pyinstaller_args.extend(["--hidden-import", imp])

    if collect_all:
        for pkg in collect_all:
            pyinstaller_args.extend(["--collect-all", pkg])

    if os.name == "nt":
        pyinstaller_args.append("--noconsole")

    pyinstaller_args.append(str(entry_file))

    pyinstaller_run(pyinstaller_args)

    if not executable_path.exists():
        raise FileNotFoundError(f"PyInstaller did not create expected executable at {executable_path}")

    return executable_path


@pytest.fixture(scope="session")
def coredinator_executable(request: pytest.FixtureRequest):
    service_name = "coredinator"
    entry_point = "coredinator/app.py"
    hidden_imports = [
        "uvicorn",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.wsproto_impl",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "uvicorn.loops",
        "uvicorn.loops.auto",
    ]
    collect_all = [
        "coredinator",
        "fastapi",
        "starlette",
        "pydantic",
    ]
    return _build_service_executable(service_name, entry_point, hidden_imports, collect_all)


@pytest.fixture(scope="session")
def coreio_executable() -> Path:
    """
    Build coreio executable using PyInstaller once per test session.

    Executable cached in test/bin/ directory for reuse across test runs.
    """
    return _build_service_executable(
        service_name="coreio",
        entry_point="coreio/main.py",
    )


@pytest.fixture(scope="session")
def corerl_executable() -> Path:
    """
    Build corerl executable using PyInstaller once per test session.

    Executable cached in test/bin/ directory for reuse across test runs.
    """
    return _build_service_executable(
        service_name="corerl",
        entry_point="corerl/main.py",
    )
