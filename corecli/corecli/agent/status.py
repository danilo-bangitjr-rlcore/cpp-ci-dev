import click

from corecli.utils.http import request


@click.command()
@click.argument("agent_id")
@click.option("--port", default=8000, help="Coredinator service port")
@click.pass_context
def status(ctx: click.Context, agent_id: str, port: int) -> None:
    """
    Show detailed status information for an agent.
    """
    data = request(f"/api/agents/{agent_id}/status", port=port, timeout=10.0).expect(
        f"Could not contact coredinator on port {port} to get agent status",
    )

    click.echo(f"Agent ID: {data['id']}")
    click.echo(f"State: {data['state']}")
    click.echo(f"Config: {data.get('config_path', 'N/A')}")

    if data.get("service_statuses"):
        click.echo("\nServices:")
        for service_name, service_status in data["service_statuses"].items():
            state = service_status.get("state", "unknown")
            click.echo(f"  {service_name}: {state}")
