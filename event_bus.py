import logging
import hydra
import asyncio

from omegaconf import DictConfig

from corerl.messages.server import WebsocketServer


async def async_main(cfg: DictConfig):
    host = cfg.host
    port = cfg.port
    server = WebsocketServer(host, port)
    await server.start()
    await server.serve_forever()


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg.event_bus))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
