#!/usr/bin/env python3

import logging
import time
import zmq

from coreio.utils.io_events import IOEventTopic, IOEventType, IOEvent

from corerl.config import MainConfig
from corerl.configs.loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect(cfg.coreio.coreio_connection)

    topic = IOEventTopic.coreio

    for _ in range(10):
        messagedata = IOEvent(type=IOEventType.write_opcua_nodes).model_dump_json()
        payload = f"{topic} {messagedata}"
        logger.info(payload)
        socket.send_string(payload)
        time.sleep(1)

    socket.close()
    context.term()

if __name__ == "__main__":
    main()


