#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging
import threading

from lib_config.loader import load_config

from coreio.communication.opc_communication import OPC_Connection_IO
from coreio.communication.scheduler import start_scheduler_io_thread
from coreio.communication.sql_communication import SQL_Manager
from coreio.communication.zmq_communication import ZMQ_Communication
from coreio.utils.config_schemas import MainConfigAdapter
from coreio.utils.event_handlers import handle_read_event, handle_write_event
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType
from coreio.utils.logging_setup import setup_logging
from coreio.utils.opc_utils import concat_opc_nodes, initialize_opc_connections

logger = setup_logging(logging.INFO)

@load_config(MainConfigAdapter)
async def coreio_loop(cfg: MainConfigAdapter):
    logger.info("Starting OPC Connections")
    # Temporary flag to keep reading opc details from pipeline in old config versions
    if cfg.coreio.data_ingress.enabled:
        opc_connections: dict[str, OPC_Connection_IO] = await initialize_opc_connections(
            cfg.coreio.opc_connections,
            cfg.coreio.tags,
            cfg.interaction.heartbeat,
        )
    else:
        opc_connections: dict[str, OPC_Connection_IO] = await initialize_opc_connections(
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
            cb=lambda event: handle_read_event(
                event,
                opc_connections,
                sql_communication,
                read_period=cfg.coreio.data_ingress.ingress_period,
            ),
        )

    logger.info("CoreIO is ready")

    try:
        async for event in zmq_communication.async_listen_forever():
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
