import click

from corecli.offline.generate_tag_config import generate_tag_config


@click.group()
def offline() -> None:
    """Offline RL training and analysis commands."""


offline.add_command(generate_tag_config)
