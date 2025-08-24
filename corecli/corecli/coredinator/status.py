import click
from rich.console import Console

from corecli.utils.coredinator import HealthModel, _check_healthcheck

console = Console()


def print_health_status(health: HealthModel, port: int) -> None:
    console.print(f"✅ Coredinator is healthy on port {port}", style="bold green")
    console.print(f"📋 Process ID: {health.process_id}", style="green")
    console.print(f"🏷️  Service: {health.service}", style="green")
    console.print(f"📦 Version: {health.version}", style="green")


def print_no_coredinator(port: int) -> None:
    console.print(f"❌ No coredinator responding on port {port}", style="bold red")


@click.command()
@click.option("--port", type=int, default=8000, help="Port of coredinator instance to check (default: 8000)")
@click.pass_context
def status(ctx: click.Context, port: int) -> None:
    """Check the health and status of the coredinator service."""
    maybe_health = _check_healthcheck(port)
    health = maybe_health.unwrap()
    if health is None:
        print_no_coredinator(port)
    else:
        print_health_status(health, port)
