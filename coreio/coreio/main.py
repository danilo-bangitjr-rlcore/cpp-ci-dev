#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import threading

from lib_config.loader import load_config

from coreio.communication.opc_communication import OPC_Connection_IO
from coreio.communication.scheduler import start_scheduler_io_thread
from coreio.communication.sql_communication import SQL_Manager
from coreio.communication.zmq_communication import ZMQ_Communication
from coreio.utils.config_schemas import InteractionConfigAdapter, MainConfigAdapter, PipelineConfigAdapter
from coreio.utils.event_handlers import handle_read_event, handle_write_event
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType
from coreio.utils.logging_setup import get_log_file_name, setup_logging
from coreio.utils.opc_utils import concat_opc_nodes, initialize_opc_connections


@load_config(MainConfigAdapter)
async def coreio_loop(cfg: MainConfigAdapter):
    log_file = get_log_file_name(cfg.coreio)
    logger = setup_logging(cfg.coreio.log_level, log_file)
    logger.info(f"Initialized logger with log file: {log_file}")
    logger.info("Starting OPC Connections")

    # Temporary flag to keep reading opc details from pipeline in old config versions
    if cfg.coreio.data_ingress.enabled:
        logger.debug("Will initialize OPC connections using coreio.tags (data_ingress enabled)")
        opc_connections: dict[str, OPC_Connection_IO] = await initialize_opc_connections(
            cfg.coreio.opc_connections,
            cfg.coreio.tags,
        )
    else:
        logger.warning("Running in compatiblity mode, data ingress disabled"
        "Will initialize OPC connections using pipeline.tags (data_ingress disabled)"
        "This feature will be deprecated soon")

        assert isinstance(cfg.pipeline, PipelineConfigAdapter), "Please define tags under pipeline.tags"

        heartbeat = None
        if isinstance(cfg.interaction, InteractionConfigAdapter):
            heartbeat = cfg.interaction.heartbeat

        opc_connections: dict[str, OPC_Connection_IO] = await initialize_opc_connections(
            cfg.coreio.opc_connections,
            cfg.pipeline.tags,
            heartbeat,
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

    logger.debug("Starting ZMQ communication thread/loop")
    zmq_communication.start()

    # Callbacks have to be async
    logger.debug("Attaching write_to_opc callback")
    zmq_communication.attach_callback(
        event_type=IOEventType.write_to_opc,
        cb=lambda event: handle_write_event(event, opc_connections),
    )

    ingress_stop_event = None
    if cfg.coreio.data_ingress.enabled:
        logger.info("Starting SQL communication - Data ingress enabled")
        logger.debug(f"Concatenating nodes from {len(opc_connections)} OPC connections")
        all_registered_nodes = concat_opc_nodes(opc_connections, skip_heartbeat=True)
        logger.debug(f"Creating SQL Manager for {len(all_registered_nodes)} nodes")
        sql_communication = SQL_Manager(
            cfg.infra,
            table_name=cfg.env.db.table_name,
            nodes_to_persist=all_registered_nodes,
        )

        ingress_stop_event = threading.Event()
        logger.debug("About to start scheduler IO thread for data ingress")
        start_scheduler_io_thread(cfg.coreio.data_ingress, ingress_stop_event, zmq_communication)

        # Callbacks have to be async
        logger.debug("Attaching read_from_opc callback")
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
        logger.debug("Starting main event loop")
        async for event in zmq_communication.async_listen_forever():
            logger.debug(f"Processing event {event.type} in main loop")
            if event.type == IOEventType.exit_io:
                logger.info("Received exit event, shutting down CoreIO...")
                break

    except Exception:
        logger.exception("CoreIO error occurred")
    finally:
        logger.info("Cleaning up OPC connections")
        logger.debug(f"Cleaning up {len(opc_connections)} OPC connections")
        for opc_conn in opc_connections.values():
            await opc_conn.cleanup()

        logger.info("Cleaning up ZMQ communication")
        zmq_communication.cleanup()
        if ingress_stop_event:
            logger.debug("Setting ingress stop event")
            ingress_stop_event.set()

        logger.info("CoreIO finished cleanup")

def main():
    asyncio.run(coreio_loop())

if __name__ == "__main__":
    main()
