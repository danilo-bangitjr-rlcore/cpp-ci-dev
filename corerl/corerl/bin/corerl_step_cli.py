#!/usr/bin/env python3

import argparse
import logging
import time

import zmq

from corerl.messages.events import Event, EventTopic, EventType
from corerl.messages.factory import EventBusConfig

_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cmd", type=EventType, required=True,
        help="See EventType for valid commnads"
    )

    parser.add_argument(
        "-b", "--bus-addr", type=str,
        default=EventBusConfig.cli_connection, help="Address of the bus for the CLI tool."
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="If specified, output will be suppressed"
    )
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s: %(message)s",
            encoding="utf-8",
            level=logging.WARNING,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s: %(message)s",
            encoding="utf-8",
            level=logging.DEBUG,
        )

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect(args.bus_addr)

    # Needed because of slow joiners:
    # Subscribers need the publisher to wait between bind and send.
    # https://zguide.zeromq.org/docs/chapter5/#Representing-State-as-Key-Value-Pairs
    time.sleep(0.2)

    topic = EventTopic.corerl_cli
    messagedata = Event(type=args.cmd).model_dump_json()
    payload = f"{topic} {messagedata}"

    socket.send_string(payload)

    _logger.info(f"Sent {payload}")

    socket.close()
    context.term()

if __name__ == "__main__":
    main()
