import click

from .start import start
from .status import status
from .stop import stop


@click.group()
def io() -> None:
    """
    CoreIO service management commands.
    """


io.add_command(start)
io.add_command(stop)
io.add_command(status)
