#!/usr/bin/env python3

import logging
import time

import numpy as np
import zmq

from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType, OPCUANodeWriteValue
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
    socket.connect(cfg.coreio.coreio_origin)
    time.sleep(0.001)

    topic = IOEventTopic.coreio

    for _ in range(30):
        # Make message
        x = np.random.rand()
        messagedata = IOEvent(
            type=IOEventType.write_opcua_nodes,
            data={"asdxf": [OPCUANodeWriteValue(node_id= "ns=2;i=2", value= x)]}
        ).model_dump_json()

        payload = f"{topic} {messagedata}"
        logger.info(payload)
        socket.send_string(payload)
        time.sleep(2)

    # Exit message
    messagedata = IOEvent(type=IOEventType.exit_io).model_dump_json()
    payload = f"{topic} {messagedata}"
    logger.info(payload)
    socket.send_string(payload)
    time.sleep(1)

    socket.close()
    context.term()

if __name__ == "__main__":
    main()
