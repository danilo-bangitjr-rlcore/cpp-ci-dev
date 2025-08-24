import os
import shutil
from pathlib import Path

from lib_utils.maybe import Maybe
from pydantic import BaseModel

from corecli.utils.daemon import start_daemon_process, stop_process_gracefully, wait_for_event
from corecli.utils.http import maybe_get_json


class CoredinatorNotFoundError(Exception):
    pass


def find_coredinator_executable() -> Path:
    def check_app_py(base_path: Path) -> Path | None:
        app_py = base_path / "coredinator" / "app.py"
        return app_py if app_py.exists() else None

    def check_executable(base_path: Path) -> Path | None:
        return base_path if base_path.is_file() and os.access(base_path, os.X_OK) else None

    def check_path_env_var():
        return (
            Maybe(shutil.which("coredinator"))
            .map(Path)
        )

    return (
        Maybe(Path.cwd() / "coredinator")
        .map(lambda p: check_app_py(p))
        .otherwise(lambda: check_executable(Path.cwd() / "coredinator"))
        .otherwise(lambda: check_executable(Path.cwd().parent / "coredinator"))
        .flat_otherwise(lambda: Maybe.flat_from_try(check_path_env_var))
        .expect(CoredinatorNotFoundError(
            "Could not find coredinator executable. Make sure coredinator is "
            "installed or run from the monorepo root.",
        ))
    )


def start_coredinator(
    port: int = 8000,
    base_path: Path | None = None,
    log_file: Path | None = None,
) -> int:
    if is_coredinator_running(port):
        pid_maybe = get_coredinator_pid(port)
        raise RuntimeError(
            f"Coredinator already running on port {port} (PID: {pid_maybe.unwrap()})",
        )

    coredinator_exe = find_coredinator_executable()

    resolved_base_path = (
        Maybe(base_path)
        .or_else(coredinator_exe.parent.parent)
    )

    command = [
        "python",
        str(coredinator_exe),
        "--base-path",
        str(resolved_base_path),
        "--port",
        str(port),
    ]

    return start_daemon_process(
        command=command,
        cwd=coredinator_exe.parent,
        log_file=log_file,
    )


def stop_coredinator(port: int = 8000, timeout: float = 10.0) -> bool:
    return (
        get_coredinator_pid(port)
        .map(lambda pid: stop_process_gracefully(pid, timeout))
        .or_else(True)
    )


class HealthModel(BaseModel):
    status: str
    process_id: int
    service: str
    version: str


def _check_healthcheck(port: int, timeout: float = 5.0) -> Maybe[HealthModel]:
    url = f"http://localhost:{port}/api/healthcheck"
    return maybe_get_json(url, HealthModel, timeout=timeout)


def is_coredinator_running(port: int = 8000) -> bool:
    return _check_healthcheck(port).is_some()


def get_coredinator_pid(port: int = 8000) -> Maybe[int]:
    return (
        _check_healthcheck(port)
        .map(lambda data: data.process_id)
    )


def wait_for_coredinator_start(port: int, timeout: float = 30.0) -> bool:
    return wait_for_event(
        predicate=lambda: is_coredinator_running(port),
        interval=0.5,
        timeout=timeout,
    )


def wait_for_coredinator_stop(port: int = 8000, timeout: float = 10.0) -> bool:
    return wait_for_event(
        predicate=lambda: not is_coredinator_running(port),
        interval=0.5,
        timeout=timeout,
    )
