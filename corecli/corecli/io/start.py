from pathlib import Path

import click

from corecli.utils.http import request


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--coreio-id", help="Custom CoreIO service ID")
@click.option("--port", default=8000, help="Coredinator service port")
@click.pass_context
def start(ctx: click.Context, config_path: str, coreio_id: str | None, port: int) -> None:
    """
    Start a new CoreIO service instance.
    """
    resolved_path = Path(config_path).resolve()

    payload = {"config_path": str(resolved_path)}
    if coreio_id:
        payload["coreio_id"] = coreio_id

    service_id = request("/api/io/start", method="POST", payload=payload, port=port).expect(
        f"Could not contact coredinator on port {port} to start CoreIO service",
    )

    click.echo(f"✅ Started CoreIO service: {service_id}")
