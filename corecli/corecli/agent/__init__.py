import click

from .list import list_agents
from .logs import logs
from .start import start
from .status import status
from .stop import stop


@click.group()
def agent() -> None:
    """
    Agent management commands.
    """


agent.add_command(start)
agent.add_command(stop)
agent.add_command(status)
agent.add_command(list_agents)
agent.add_command(logs)
