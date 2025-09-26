import click


@click.command()
@click.pass_context
def stop(ctx: click.Context) -> None:
    """
    Gracefully stop the coredinator service.
    """
    # Implementation will be added later
    click.echo("🚧 Stopping coredinator...")
