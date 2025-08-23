import click


@click.command("stop-sim")
@click.option("--clean", is_flag=True, help="Stop and remove volumes")
@click.pass_context
def stop_sim(ctx: click.Context, clean: bool) -> None:
    """
    Cleanly shut down the simulation environment.
    """
    # Implementation will be added later
    click.echo("🚧 Stopping simulation environment")
    if clean:
        click.echo("🧹 Cleaning up volumes")
