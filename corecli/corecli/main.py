#!/usr/bin/env python3
import logging
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

log = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CoreCLI - Command line tools"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Verbose mode enabled")


@cli.command()
@click.option('--name', '-n', default='Developer', help='Name to greet')
@click.pass_context
def hello(ctx: click.Context, name: str) -> None:
    console.print(f"👋 Hello, {name}!", style="bold green")
    console.print("Welcome to CoreCLI - your development companion!")

    if ctx.obj.get('verbose'):
        console.print(f"[dim]Debug: Greeting user '{name}'[/dim]")


@cli.command()
def version() -> None:
    from corecli import __version__
    console.print(f"CoreCLI version: {__version__}", style="bold blue")


def main() -> Any:
    return cli()


if __name__ == '__main__':
    main()
