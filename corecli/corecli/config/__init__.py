import click

from .validate import validate


@click.group()
def config() -> None:
    """
    Configuration management commands.
    """


config.add_command(validate)
