import click


@click.command()
@click.option("--port", default=8080, help="Port to start coredinator on")
@click.pass_context
def start(ctx: click.Context, port: int) -> None:
    """
    Start the coredinator service if not running.
    """
    # Implementation will be added later
    click.echo(f"🚧 Starting coredinator on port {port}...")
