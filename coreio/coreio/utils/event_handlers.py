import logging

from coreio.communication.opc_communication import OPC_Connection
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.io_events import IOEvent

logger = logging.getLogger()

async def handle_write_event(event: IOEvent, opc_connections: dict[str, OPC_Connection]):
    logger.info(f"Received writing event {event}")
    for connection_id, payload in event.data.items():
        opc_conn = opc_connections.get(connection_id)
        if opc_conn is None:
            logger.warning(f"Connection Id {connection_id} is unknown.")
            continue

        async with opc_conn:
            await opc_conn.write_opcua_nodes(payload)

async def handle_read_event(event: IOEvent, opc_connections: dict[str, OPC_Connection], sql_communication: SQL_Manager):
    logger.info(f"Received reading event {event}")

    nodes_name_val = {}

    for opc_conn in opc_connections.values():
        async with opc_conn:
            nodes_name_val = nodes_name_val | await opc_conn.read_nodes_named(opc_conn.registered_nodes)

    logger.info(f"Read nodes value: {nodes_name_val}")

    if not nodes_name_val:
        logger.warning("No node values read; skipping SQL write.")
        return

    try:
        sql_communication.write_nodes(nodes_name_val, event.time)
    except Exception as exc:
        logger.error(f"Failed to write nodes to SQL: {exc}")
