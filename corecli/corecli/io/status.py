import click

from corecli.utils.http import request


@click.command()
@click.argument("service_id")
@click.option("--port", default=8000, help="Coredinator service port")
@click.pass_context
def status(ctx: click.Context, service_id: str, port: int) -> None:
    """
    Show detailed status information for a CoreIO service.
    """
    data = request(f"/api/io/{service_id}/status", port=port, timeout=10.0).expect(
        f"Could not contact coredinator on port {port} to get CoreIO status",
    )

    click.echo(f"Service ID: {data['service_id']}")

    if "status" in data:
        status_data = data["status"]
        click.echo(f"State: {status_data.get('state', 'unknown')}")
        click.echo(f"Config: {status_data.get('config_path', 'N/A')}")
