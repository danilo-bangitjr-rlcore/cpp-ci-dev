import click


@click.command("list")
@click.option("--running", is_flag=True, help="Show only running agents")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
@click.pass_context
def list_agents(ctx: click.Context, running: bool, verbose: bool) -> None:
    """
    List all agents known to coredinator.
    """
    # Implementation will be added later
    click.echo("🚧 Listing agents...")
    if running:
        click.echo("🔄 Showing only running agents")
    if verbose:
        click.echo("📋 Verbose output requested")
