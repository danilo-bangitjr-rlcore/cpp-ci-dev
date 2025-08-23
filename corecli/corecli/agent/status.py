import click


@click.command()
@click.argument("config_path_or_agent_id")
@click.pass_context
def status(ctx: click.Context, config_path_or_agent_id: str) -> None:
    """
    Show detailed status information for an agent.
    """
    # Implementation will be added later
    click.echo(f"🚧 Checking status for agent: {config_path_or_agent_id}")
