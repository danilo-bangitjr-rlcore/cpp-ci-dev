import click

from .logs import logs
from .start_sim import start_sim
from .stop_sim import stop_sim


@click.group()
def dev() -> None:
    """
    Development workflow commands.
    """


dev.add_command(start_sim)
dev.add_command(stop_sim)
dev.add_command(logs)
