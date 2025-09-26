import click

from .start import start
from .status import status
from .stop import stop


@click.group()
def coredinator() -> None:
    """
    Coredinator management commands.
    """


coredinator.add_command(status)
coredinator.add_command(start)
coredinator.add_command(stop)
