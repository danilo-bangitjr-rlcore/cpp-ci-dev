import click


@click.command("start-sim")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--clean", is_flag=True, help="Clean start (remove existing containers)")
@click.pass_context
def start_sim(ctx: click.Context, config_path: str, clean: bool) -> None:
    """
    Start development simulation environment with all required services.
    """
    # Implementation will be added later
    click.echo(f"🚧 Starting simulation with config: {config_path}")
    if clean:
        click.echo("🧹 Clean start requested")
