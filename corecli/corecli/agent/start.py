from pathlib import Path

import click

from corecli.utils.http import request


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--coreio-id", help="Shared CoreIO service ID to use")
@click.option("--port", default=8000, help="Coredinator service port")
@click.option("--follow-logs", is_flag=True, help="Follow agent logs after starting")
@click.pass_context
def start(
    ctx: click.Context,
    config_path: str,
    coreio_id: str | None,
    port: int,
    follow_logs: bool,
) -> None:
    """
    Start a new RL agent using the specified configuration.
    """
    resolved_path = Path(config_path).resolve()

    payload = {"config_path": str(resolved_path)}
    if coreio_id:
        payload["coreio_id"] = coreio_id

    agent_id = request("/api/agents/start", method="POST", payload=payload, port=port).expect(
        f"Could not contact coredinator on port {port} to start agent",
    )

    click.echo(f"✅ Started agent: {agent_id}")

    if coreio_id:
        click.echo(f"🔗 Using shared CoreIO: {coreio_id}")

    if follow_logs:
        click.echo(f"📝 To follow logs, run: corecli agent logs {agent_id}")
