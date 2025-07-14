#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging

import colorlog
from lib_config.loader import load_config

from coreio.communication.opc_communication import OPC_Connection
from coreio.communication.sql_communication import SQL_Manager
from coreio.communication.zmq_communication import ZMQ_Communication
from coreio.utils.config_schemas import MainConfigAdapter
from coreio.utils.io_events import IOEventType
from coreio.utils.opc_utils import concat_opc_nodes

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
    opc_connections: dict[str, OPC_Connection] = {}
    for opc_conn_cfg in cfg.coreio.opc_connections:
        logger.info(f"Connecting to OPC Connection {opc_conn_cfg.connection_id} at {opc_conn_cfg.opc_conn_url}")
        opc_conn = await OPC_Connection().init(opc_conn_cfg)
        opc_connections[opc_conn_cfg.connection_id] = opc_conn

        async with opc_conn:
            await opc_conn.register_cfg_nodes(cfg.pipeline.tags, ai_setpoint_only = True)

        # Register heartbeat_id separately
        if cfg.interaction.heartbeat.connection_id == opc_conn_cfg.connection_id:
            heartbeat_id = cfg.interaction.heartbeat.heartbeat_node_id

            if heartbeat_id is not None:
                async with opc_conn:
                    await opc_conn.register_node(heartbeat_id, "heartbeat")

    if cfg.coreio.data_ingress.enabled:
        logger.info("Starting SQL communication")
        all_registered_nodes = concat_opc_nodes(opc_connections, skip_heartbeat=True)
        sql_communication = SQL_Manager(
            cfg.infra,
            table_name=cfg.env.db.table_name,
            nodes_to_persist=all_registered_nodes,
        )
        print(sql_communication)

    logger.info("Starting ZMQ communication")
    zmq_communication = ZMQ_Communication(cfg.coreio)
    zmq_communication.start()

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

                case IOEventType.exit_io:
                    logger.info("Received exit event, shutting down CoreIO...")
                    break
    except Exception:
        logger.exception("CoreIO error occurred")
    finally:
        zmq_communication.cleanup()
        logger.info("CoreIO finished cleanup")

def main():
    asyncio.run(coreio_loop())

if __name__ == "__main__":
    main()

