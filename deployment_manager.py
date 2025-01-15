import asyncio
import datetime as dt
from dataclasses import field

from corerl.configs.config import MISSING, config, list_
from corerl.configs.loader import load_config
from corerl.messages.client import WebsocketClient
from corerl.messages.events import EventType
from corerl.messages.server import WebsocketServerConfig
from corerl.utils.processes import keep_alive


@config()
class EventBusDeploymentConfig(WebsocketServerConfig):
    python_entrypoint: str = MISSING
    base: str = MISSING
    config_name: str = MISSING


@config()
class DeploymentConfig:
    python_executable: str = MISSING
    python_entrypoint: str = MISSING
    base: str = MISSING
    config_name: str = MISSING
    options: list[str] = list_()


@config()
class DeploymentMangerConfig:
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    event_bus: EventBusDeploymentConfig = field(default_factory=EventBusDeploymentConfig)


def parse_python_executable(cfg: DeploymentMangerConfig):
    # the python executable could actually be a nested command, such as:
    #    uv run python
    # where we want to pass
    #    [ 'uv', 'run', 'python' ]
    # to subprocess
    return (
        str(cfg.deployment.python_executable)
        .removeprefix('"')
        .removesuffix('"')
        .split(' ')
    )


async def start_agent(cfg: DeploymentMangerConfig):
    exec = parse_python_executable(cfg)
    python_entrypoint = cfg.deployment.python_entrypoint
    config_name = cfg.deployment.config_name
    options = cfg.deployment.options

    cmd = [*exec, python_entrypoint, '--config-name', config_name, *options]

    await keep_alive(
        cmd,
        base_backoff=2,
        max_backoff=dt.timedelta(hours=1),
    )


async def start_event_bus(cfg: DeploymentMangerConfig):
    exec = parse_python_executable(cfg)
    python_entrypoint = cfg.event_bus.python_entrypoint

    await keep_alive(
        cmd=[*exec, python_entrypoint, '--base', cfg.event_bus.base, '--config-name', cfg.event_bus.config_name],
        base_backoff=2,
        max_backoff=dt.timedelta(minutes=5),
    )


async def start_event_client(cfg: DeploymentMangerConfig):
    host = cfg.event_bus.host
    port = cfg.event_bus.port

    client = WebsocketClient(host, port)
    await client.start()
    await client.ensure_connected()
    return client


async def async_main(cfg: DeploymentMangerConfig):
    # give the event bus a brief headstart before
    # kicking off other services.
    # We know the bus has successfully started if we can
    # connect a client to it.
    eb_future = asyncio.ensure_future(start_event_bus(cfg))

    await asyncio.sleep(1)
    client = await start_event_client(cfg)
    asyncio.ensure_future(client.listen_forever())

    await client.subscribe(
        to=EventType.agent_get_action,
        cb=print,
    )

    await start_agent(cfg)
    eb_future.cancel()


@load_config(DeploymentMangerConfig)
def main(cfg: DeploymentMangerConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg))


if __name__ == '__main__':
    main()
