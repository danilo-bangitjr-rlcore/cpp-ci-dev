import click


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--name", help="Custom name for the agent")
@click.option("--follow-logs", is_flag=True, help="Follow agent logs after starting")
@click.pass_context
def start(ctx: click.Context, config_path: str, name: str | None, follow_logs: bool) -> None:
    """
    Start a new RL agent using the specified configuration.
    """
    # Implementation will be added later
    click.echo(f"🚧 Starting agent with config: {config_path}")

    if name is not None:
        click.echo(f"🏷️  Agent name: {name}")

    if follow_logs:
        click.echo("📝 Will follow logs after start")
