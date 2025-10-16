import click

from corecli.utils.http import request


@click.command()
@click.argument("agent_id")
@click.option("--port", default=8000, help="Coredinator service port")
@click.option("--force", is_flag=True, help="Force stop if graceful shutdown fails")
@click.pass_context
def stop(ctx: click.Context, agent_id: str, port: int, force: bool) -> None:
    """
    Stop a running agent.
    """
    request(f"/api/agents/{agent_id}/stop", method="POST", port=port).expect(
        f"Could not contact coredinator on port {port} to stop agent {agent_id}",
    )

    click.echo(f"✅ Stopped agent: {agent_id}")

    if force:
        click.echo("⚡ Force stop requested (not yet implemented)")
