import click


@click.command()
@click.argument("config_path_or_agent_id")
@click.option("--force", is_flag=True, help="Force stop if graceful shutdown fails")
@click.pass_context
def stop(ctx: click.Context, config_path_or_agent_id: str, force: bool) -> None:
    """
    Stop a running agent.
    """
    # Implementation will be added later
    click.echo(f"🚧 Stopping agent: {config_path_or_agent_id}")
    if force:
        click.echo("⚡ Force stop requested")
