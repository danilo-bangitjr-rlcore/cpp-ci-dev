#!/usr/bin/env python3
import logging
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler

from corecli.agent import agent
from corecli.config import config
from corecli.coredinator import coredinator
from corecli.dev import dev
from corecli.offline import offline

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

log = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.option("--config-dir", help="Override default config directory")
@click.option("--coredinator-url", help="Override default coredinator URL")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config_dir: str | None, coredinator_url: str | None) -> None:
    """
    CoreCLI - Command line tools for the engineering team.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config_dir"] = config_dir
    ctx.obj["coredinator_url"] = coredinator_url

    # Configure logging based on verbosity
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Verbose mode enabled")

    # Log configuration
    if config_dir is not None:
        log.debug(f"Using config directory: {config_dir}")

    if coredinator_url is not None:
        log.debug(f"Using coredinator URL: {coredinator_url}")


@cli.command()
def version() -> None:
    """
    Show the version of corecli.
    """
    from corecli import __version__

    console.print(f"CoreCLI version: {__version__}", style="bold blue")


# Register command groups
cli.add_command(dev)
cli.add_command(coredinator)
cli.add_command(agent)
cli.add_command(config)
cli.add_command(offline)


def main() -> Any:
    return cli()


if __name__ == "__main__":
    main()
