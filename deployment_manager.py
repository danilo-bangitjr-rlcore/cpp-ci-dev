import hydra
import asyncio
import datetime as dt

from omegaconf import DictConfig
from corerl.messages.client import WebsocketClient
from corerl.messages.events import EventType
from corerl.utils.processes import keep_alive


def parse_python_executable(cfg: DictConfig):
    # the python executable could actually be a nested command, such as:
    #    uv run python
    # where we want to pass
    #    [ 'uv', 'run', 'python' ]
    # to subprocess
    return str(cfg.deployment.python_executable).split(' ')


async def start_agent(cfg: DictConfig):
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


async def start_event_bus(cfg: DictConfig):
    exec = parse_python_executable(cfg)
    python_entrypoint = cfg.event_bus.python_entrypoint

    await keep_alive(
        cmd=[*exec, python_entrypoint],
        base_backoff=2,
        max_backoff=dt.timedelta(minutes=5),
    )


async def start_event_client(cfg: DictConfig):
    host = cfg.event_bus.host
    port = cfg.event_bus.port

    client = WebsocketClient(host, port)
    await client.start()
    await client.ensure_connected()
    return client


async def async_main(cfg: DictConfig):
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


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg))


if __name__ == '__main__':
    main()
