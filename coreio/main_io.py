#!/usr/bin/env python3
# CoreIO is async first

import asyncio
import logging

from coreio.utils.io_events import IOEvent, IOEventType
from coreio.utils.opc_communication import OPC_Connection
from coreio.utils.zmq_communication import ZMQ_Communication
from corerl.config import MainConfig
from corerl.configs.loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@load_config(MainConfig, base='config/')
async def coreio_loop(cfg: MainConfig):
    opc_connections: dict[str, OPC_Connection] = {}
    for opc_conn_cfg in cfg.coreio.opc_connections:
        opc_connections[opc_conn_cfg.connection_id] = await OPC_Connection().init(opc_conn_cfg, cfg.pipeline.tags)

        # Register heartbeat_id separately
        if cfg.interaction.heartbeat.connection_id == opc_conn_cfg.connection_id:
            heartbeat_id = cfg.interaction.heartbeat.heartbeat_node_id

            if heartbeat_id is not None:
                await opc_connections[opc_conn_cfg.connection_id].register_node(heartbeat_id)

    for opc_conn in opc_connections.values():
        await opc_conn.start()

    zmq_communication = ZMQ_Communication(cfg.coreio)
    zmq_communication.start()

    logger.info("CoreIO is ready")

    while True:
        event = zmq_communication.recv_event()

        if event is None:
            return

        match event.type:
            case IOEventType.write_opcua_nodes:
                for connection_id, payload in event.data.items():
                    opc_conn = opc_connections.get(connection_id)
                    if opc_conn is None:
                        logger.warning(f"Connection Id {connection_id} is unkown.")
                        continue

                    await opc_conn.write_opcua_nodes(payload)

            case IOEventType.exit_io:
                break

    zmq_communication.cleanup()
    for opc_conn in opc_connections.values():
        await opc_conn.cleanup()

    logger.info("CoreIO finished cleanup")

def main():
    asyncio.run(coreio_loop())

if __name__ == "__main__":
    main()
