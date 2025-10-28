import click

from corecli.offline.bclone import bclone
from corecli.offline.data_report import data_report
from corecli.offline.generate_tag_config import generate_tag_config
from corecli.offline.train import train
from corecli.offline.trajectory_report import trajectory_report


@click.group()
def offline() -> None:
    """Offline RL training and analysis commands."""


offline.add_command(generate_tag_config)
offline.add_command(train)
offline.add_command(bclone)
offline.add_command(data_report)
offline.add_command(trajectory_report)
