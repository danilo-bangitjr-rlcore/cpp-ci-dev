import logging
from datetime import UTC, datetime, timedelta

from asyncua.server.address_space import NodeData

from coreio.communication.opc_communication import OPC_Connection_IO
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.io_events import IOEvent

logger = logging.getLogger()

STALE_EVENT_THRESHOLD_MULTIPLIER = 3

def is_stale_event(event: IOEvent, max_stale: timedelta = timedelta(seconds=5)):
    logger.debug(f"Checking if event from {event.time} is stale (max_stale: {max_stale})")
    event_time = datetime.fromisoformat(event.time)
    if (datetime.now(UTC) - event_time) > max_stale:
        return True
    return False


async def handle_write_event(event: IOEvent, opc_connections: dict[str, OPC_Connection_IO]):
    logger.info(f"Received writing event {event}")
    logger.debug(f"Processing write event for {len(event.data)} connections")
    for connection_id, payload in event.data.items():
        logger.debug(f"About to write {len(payload)} nodes to connection {connection_id}")
        opc_conn = opc_connections.get(connection_id)
        if opc_conn is None:
            logger.warning(f"Connection Id {connection_id} is unknown.")
            continue

        try:
            await opc_conn.write_opcua_nodes(payload)
        except Exception as exc:
            logger.error(f"Failed to write nodes to OPC: {exc!s}")

async def handle_read_event(
        event: IOEvent,
        opc_connections: dict[str, OPC_Connection_IO],
        sql_communication: SQL_Manager,
        read_period: timedelta,
):
    logger.info(f"Received reading event {event}")
    logger.debug(f"Checking event staleness with threshold {read_period * STALE_EVENT_THRESHOLD_MULTIPLIER}")
    if is_stale_event(event, read_period * STALE_EVENT_THRESHOLD_MULTIPLIER):
        logger.warning(f"Dropping {event} because it is stale")
        return

    logger.debug(f"About to read nodes from {len(opc_connections)} OPC connections")
    nodes_name_val: dict[str, NodeData] = {}

    for opc_conn in opc_connections.values():
        try:
            logger.debug(f"Reading {len(opc_conn.registered_nodes)} registered nodes from {opc_conn.connection_id}")
            nodes_name_val = nodes_name_val | await opc_conn.read_nodes_named(opc_conn.registered_nodes)
        except Exception as exc:
            logger.error(f"Failed to read nodes from OPC: {exc!s}")

    logger.info(f"Read nodes value: {nodes_name_val}")

    if not nodes_name_val:
        logger.warning("No node values read; skipping SQL write.")
        return

    logger.debug(f"About to write {len(nodes_name_val)} node values to SQL")
    try:
        sql_communication.write_to_sql(nodes_name_val, event.time)
    except Exception as exc:
        logger.error(f"Failed to write nodes to SQL: {exc!s}")
