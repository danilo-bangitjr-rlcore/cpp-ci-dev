import logging

import click
from rich.console import Console

from corecli.utils import stop_coredinator, wait_for_coredinator_stop
from corecli.utils.cli import handle_exceptions

console = Console()
log = logging.getLogger(__name__)


@click.command()
@click.option("--port", type=int, default=8000, help="Port of coredinator instance to stop (default: 8000)")
@click.option("--timeout", default=10.0, help="Seconds to wait for shutdown")
@click.pass_context
@handle_exceptions()
def stop(ctx: click.Context, port: int, timeout: float) -> None:
    """Gracefully stop the coredinator service."""
    console.print("🛑 Stopping coredinator...", style="blue")

    # Stop the service
    success = stop_coredinator(port, timeout)

    if not success:
        console.print("❌ Failed to stop some coredinator processes", style="bold red")
        ctx.exit(1)

    console.print("📋 Coredinator stop signal sent", style="green")

    # Wait for it to actually stop
    console.print("⏳ Waiting for coredinator to stop...", style="yellow")

    if wait_for_coredinator_stop(port, timeout):
        console.print("✅ Coredinator stopped successfully", style="bold green")
    else:
        console.print(f"⚠️  Coredinator may still be running (timeout after {timeout}s)", style="yellow")
        console.print("Check with 'corecli coredinator status'", style="dim")
