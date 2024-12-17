import logging
import asyncio

from corerl.configs.loader import load_config
from corerl.messages.server import WebsocketServer, WebsocketServerConfig


async def async_main(cfg: WebsocketServerConfig):
    server = WebsocketServer(cfg)
    await server.start()
    await server.serve_forever()


@load_config(WebsocketServerConfig, base='config/')
def main(cfg: WebsocketServerConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
