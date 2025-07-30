#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging
import threading

import colorlog
from lib_config.loader import load_config
from lib_utils.messages.base_event_bus import BaseEventBus

from coreio.communication.opc_communication import OPC_Connection
from coreio.communication.scheduler import start_scheduler_io_thread
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.config_schemas import MainConfigAdapter
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType
from coreio.utils.opc_utils import concat_opc_nodes, initialize_opc_connections

colorlog.basicConfig(
    level=logging.DEBUG,
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
    zmq_communication = BaseEventBus(
        event_class=IOEvent,
        topic=IOEventTopic.coreio,
        consumer_name="coreio_consumer",
        subscriber_addrs=[cfg.coreio.coreio_origin, cfg.coreio.coreio_app],
        publisher_addr=cfg.coreio.coreio_app,
    )
    zmq_communication.start()

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

    logger.info("CoreIO is ready")

    try:
        while True:
            event = zmq_communication.recv_event()
            if event is None:
                continue

            match event.type:
                case IOEventType.write_opcua_nodes:
                    logger.info(f"Received writing event {event}")
                    for connection_id, payload in event.data.items():
                        opc_conn = opc_connections.get(connection_id)
                        if opc_conn is None:
                            logger.warning(f"Connection Id {connection_id} is unkown.")
                            continue

                        async with opc_conn:
                            await opc_conn.write_opcua_nodes(payload)

                case IOEventType.read_opcua_nodes:
                    logger.info(f"Received reading event {event}")

                    if sql_communication is None:
                        logger.error("SQL Communication must be enabled to handle read events")
                        continue

                    nodes_name_val = {}

                    for opc_conn in opc_connections.values():
                        async with opc_conn:
                            nodes_name_val = nodes_name_val | await opc_conn.read_nodes_named(opc_conn.registered_nodes)

                    logger.info(f"Read nodes value: {nodes_name_val}")

                    if not nodes_name_val:
                        logger.warning("No node values read; skipping SQL write.")
                        continue

                    try:
                        sql_communication.write_nodes(nodes_name_val, event.time)
                    except Exception as exc:
                        logger.error(f"Failed to write nodes to SQL: {exc}")

                case IOEventType.exit_io:
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

