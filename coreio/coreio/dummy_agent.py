#!/usr/bin/env python3

import logging
import time

import numpy as np
import zmq
from lib_config.loader import load_config

from coreio.utils.config_schemas import MainConfigAdapter
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType, OPCUANodeWriteValue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

@load_config(MainConfigAdapter)
def main(cfg: MainConfigAdapter):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect(cfg.coreio.coreio_origin)
    time.sleep(0.001)

    topic = IOEventTopic.coreio
    node_id = cfg.pipeline.tags[0].node_identifier
    action_period = cfg.interaction.action_period.total_seconds()
    assert node_id is not None, "No tags in config.yaml"

    for i in range(30):
        # Make message
        x = np.random.rand()
        if i % 3 == 1:
            x = None
        if i % 3 == 2:
            x = np.nan

        # Send write opc event
        messagedata = IOEvent(
            type=IOEventType.write_to_opc,
            data={"asdxf": [OPCUANodeWriteValue(node_id=node_id, value= x)]},
        ).model_dump_json()

        payload = f"{topic} {messagedata}"
        logger.info(payload)
        socket.send_string(payload)

        if cfg.coreio.data_ingress.enabled:
            # Send read from opc event
            messagedata = IOEvent(
                type = IOEventType.read_from_opc,
                data={},
            ).model_dump_json()

            payload = f"{topic} {messagedata}"
            logger.info(payload)
            socket.send_string(payload)

        time.sleep(action_period)

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
