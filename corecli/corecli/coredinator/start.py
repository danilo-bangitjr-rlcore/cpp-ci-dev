import logging
from pathlib import Path

import click
from rich.console import Console

from corecli.utils import (
    CoredinatorNotFoundError,
    start_coredinator,
    wait_for_coredinator_start,
)
from corecli.utils.cli import handle_exceptions

console = Console()
log = logging.getLogger(__name__)


@click.command()
@click.option("--port", default=8000, help="Port to start coredinator on")
@click.option(
    "--base-path",
    type=click.Path(path_type=Path, file_okay=False, resolve_path=True),
    help="Path to microservice executables",
)
@click.option("--log-file", type=click.Path(path_type=Path), help="Log file for coredinator output")
@click.option("--timeout", default=30.0, help="Seconds to wait for startup")
@click.pass_context
@handle_exceptions(
    {
        CoredinatorNotFoundError: "Coredinator executable not found",
        RuntimeError: "Runtime error during startup",
    },
)
def start(
    ctx: click.Context,
    port: int,
    base_path: Path | None,
    log_file: Path | None,
    timeout: float,
) -> None:
    """Start the coredinator service as a daemon process."""
    console.print(f"🚀 Starting coredinator on port {port}...", style="blue")

    if base_path is not None and base_path.exists() and not base_path.is_dir():
        raise click.BadParameter("Base path must be a directory.", param_hint="--base-path")

    # Start the daemon
    pid = start_coredinator(
        port=port,
        base_path=base_path,
        log_file=log_file,
    )

    console.print(f"📋 Coredinator started with PID {pid}", style="green")

    # Wait for it to be ready
    console.print("⏳ Waiting for coredinator to be ready...", style="yellow")

    if wait_for_coredinator_start(port, timeout):
        console.print(f"✅ Coredinator is ready on port {port}", style="bold green")
        if log_file:
            console.print(f"📝 Logs: {log_file}", style="dim")
    else:
        console.print(f"⚠️  Coredinator may not be fully ready (timeout after {timeout}s)", style="yellow")
        console.print("Check the logs for more information", style="dim")
