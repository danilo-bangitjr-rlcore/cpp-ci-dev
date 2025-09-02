import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from coreio.communication.opc_communication import OPC_Connection_IO
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.io_events import IOEvent

logger = logging.getLogger()

STALE_EVENT_THRESHOLD_MULTIPLIER = 3

def is_stale_event(event: IOEvent, max_stale: timedelta = timedelta(seconds=5)):
    event_time = datetime.fromisoformat(event.time)
    if (datetime.now(UTC) - event_time) > max_stale:
        return True
    return False


async def handle_write_event(event: IOEvent, opc_connections: dict[str, OPC_Connection_IO]):
    logger.info(f"Received writing event {event}")
    for connection_id, payload in event.data.items():
        opc_conn = opc_connections.get(connection_id)
        if opc_conn is None:
            logger.warning(f"Connection Id {connection_id} is unknown.")
            continue

        async with opc_conn:
            await opc_conn.write_opcua_nodes(payload)

async def handle_read_event(
        event: IOEvent,
        opc_connections: dict[str, OPC_Connection_IO],
        sql_communication: SQL_Manager,
        read_period: timedelta,
):
    logger.info(f"Received reading event {event}")
    if is_stale_event(event, read_period * STALE_EVENT_THRESHOLD_MULTIPLIER):
        logger.warning(f"Dropping {event} because it is stale")
        return

    nodes_name_val: dict[str, Any] = {}

    for opc_conn in opc_connections.values():
        async with opc_conn:
            nodes_name_val = nodes_name_val | await opc_conn.read_nodes_named(opc_conn.registered_nodes)

    logger.info(f"Read nodes value: {nodes_name_val}")

    if not nodes_name_val:
        logger.warning("No node values read; skipping SQL write.")
        return

    try:
        sql_communication.write_to_sql(nodes_name_val, event.time)
    except Exception as exc:
        logger.error(f"Failed to write nodes to SQL: {exc}")
