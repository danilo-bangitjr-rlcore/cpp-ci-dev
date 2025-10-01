import click


@click.group()
def offline() -> None:
    """Offline RL training and analysis commands."""
