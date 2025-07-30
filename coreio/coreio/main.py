#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging
import threading

import colorlog
from lib_config.loader import load_config

from coreio.communication.opc_communication import OPC_Connection
from coreio.communication.scheduler import start_scheduler_io_thread
from coreio.communication.sql_communication import SQL_Manager
from coreio.communication.zmq_communication import ZMQ_Communication
from coreio.utils.config_schemas import MainConfigAdapter
from coreio.utils.event_handlers import handle_read_event, handle_write_event
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType
from coreio.utils.opc_utils import concat_opc_nodes, initialize_opc_connections

colorlog.basicConfig(
    level=logging.INFO, # Can change this manually
    format='%(log_color)s%(levelname)s%(reset)s: %(asctime)s %(message)s',
    datefmt= '%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger("asyncua").setLevel(logging.WARNING)
logging.getLogger("asyncuagds").setLevel(logging.WARNING)

@load_config(MainConfigAdapter)
async def coreio_loop(cfg: MainConfigAdapter):
    logger.info("Starting OPC Connections")
    opc_connections: dict[str, OPC_Connection] = await initialize_opc_connections(
        cfg.coreio.opc_connections,
        cfg.pipeline.tags,
        cfg.interaction.heartbeat,
    )

    all_registered_nodes = None
    sql_communication = None

    logger.info("Starting ZMQ communication")
    zmq_communication = ZMQ_Communication(
        event_class=IOEvent,
        topic=IOEventTopic.coreio,
        consumer_name="coreio_consumer",
        subscriber_addrs=[cfg.coreio.coreio_origin, cfg.coreio.coreio_app],
        publisher_addr=cfg.coreio.coreio_app,
    )

    zmq_communication.start()

    # Callbacks have to be async
    zmq_communication.attach_callback(
        event_type=IOEventType.write_to_opc,
        cb=lambda event: handle_write_event(event, opc_connections),
    )

    ingress_stop_event = None
    if cfg.coreio.data_ingress.enabled:
        logger.info("Starting SQL communication")
        all_registered_nodes = concat_opc_nodes(opc_connections, skip_heartbeat=True)
        sql_communication = SQL_Manager(
            cfg.infra,
            table_name=cfg.env.db.table_name,
            nodes_to_persist=all_registered_nodes,
        )

        ingress_stop_event = threading.Event()
        start_scheduler_io_thread(cfg.coreio.data_ingress, ingress_stop_event, zmq_communication)

        # Callbacks have to be async
        zmq_communication.attach_callback(
            event_type=IOEventType.read_from_opc,
            cb=lambda event: handle_read_event(event, opc_connections, sql_communication),
        )

    event_stream = zmq_communication.async_listen_forever()
    logger.info("CoreIO is ready")

    try:
        async for event in event_stream:
            if event.type == IOEventType.exit_io:
                logger.info("Received exit event, shutting down CoreIO...")
                break

    except Exception:
        logger.exception("CoreIO error occurred")
    finally:
        zmq_communication.cleanup()
        if ingress_stop_event:
            ingress_stop_event.set()

        logger.info("CoreIO finished cleanup")

def main():
    asyncio.run(coreio_loop())

if __name__ == "__main__":
    main()
