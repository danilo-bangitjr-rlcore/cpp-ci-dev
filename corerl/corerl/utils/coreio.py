"""
corerl.utils.coreio
~~~~~~~~~~~~~~~~~~~

This module implements the CoreRL communications to CoreIO (ThinClient) process,
motivated by the separation of responsibilites between the input/output code and the agent code.
"""

import logging
import time

import zmq

from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType, OPCUANodeWriteValue

logger = logging.getLogger(__name__)

class CoreIOLink:
    def __init__(self, coreio_origin: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(coreio_origin)
        self.topic = IOEventTopic.coreio
        time.sleep(0.05)

    def write_opcua_nodes(self, nodes_to_write: dict[str, list[OPCUANodeWriteValue]]):
        message_data = IOEvent(
            type=IOEventType.write_opcua_nodes,
            data=nodes_to_write
        ).model_dump_json()
        payload = f"{self.topic} {message_data}"
        logger.info(payload)
        self.socket.send_string(payload)

    def cleanup(self):

        # Tell CoreIO to exit
        messagedata = IOEvent(type=IOEventType.exit_io).model_dump_json()
        payload = f"{self.topic} {messagedata}"
        logger.info(payload)
        self.socket.send_string(payload)
        time.sleep(0.05)

        self.socket.close()
        self.context.term()
