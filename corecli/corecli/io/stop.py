import click

from corecli.utils.http import request


@click.command()
@click.argument("service_id")
@click.option("--port", default=8000, help="Coredinator service port")
@click.pass_context
def stop(ctx: click.Context, service_id: str, port: int) -> None:
    """
    Stop a CoreIO service instance.
    """
    request(f"/api/io/{service_id}/stop", method="POST", port=port).expect(
        f"Could not contact coredinator on port {port} to stop CoreIO service {service_id}",
    )

    click.echo(f"✅ Stopped CoreIO service: {service_id}")
