from dataclasses import dataclass
import logging
import hydra
import asyncio

from hydra.core.config_store import ConfigStore
from corerl.messages.server import WebsocketServer, WebsocketServerConfig


async def async_main(cfg: WebsocketServerConfig):
    server = WebsocketServer(cfg)
    await server.start()
    await server.serve_forever()


@dataclass
class MainConfig:
    event_bus: WebsocketServerConfig


cs = ConfigStore.instance()
cs.store(name='base_config', node=MainConfig)

@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: MainConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg.event_bus))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
