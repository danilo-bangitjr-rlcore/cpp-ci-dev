import logging
import threading
from datetime import timedelta
from threading import Thread

from coreio.utils.io_events import OPCUANodeWriteValue

from corerl.configs.config import config
from corerl.utils.coreio import CoreIOLink
from corerl.utils.time import clock_generator, wait_for_timestamp

logger = logging.getLogger(__name__)


@config()
class HeartbeatConfig:
    connection_id: str | None = None
    heartbeat_node_id: str | None = None
    heartbeat_period: timedelta = timedelta(seconds=5)
    max_counter: int = 1000


def heartbeat(cfg: HeartbeatConfig, coreio_origin: str):
    if cfg.heartbeat_node_id is None:
        return

    if cfg.connection_id is None:
        return

    connection_id = cfg.connection_id
    heartbeat_node_id = cfg.heartbeat_node_id
    heartbeat_clock = clock_generator(tick_period=cfg.heartbeat_period)
    heartbeat_counter = 0

    while True:
        def _beat(counter: int):
            # initialize client on every beat
            coreio_client = CoreIOLink(coreio_origin)

            try:
                coreio_client.write_opcua_nodes({
                    connection_id: [OPCUANodeWriteValue(node_id=heartbeat_node_id, value=counter)]
                })
            except Exception:
                logger.exception("Heartbeat failed to write to coreio")

            # wait for next heartbeat
            next_heartbeat_ts = next(heartbeat_clock)
            wait_for_timestamp(next_heartbeat_ts)

        _beat(heartbeat_counter)
        # increment counter
        heartbeat_counter += 1
        heartbeat_counter %= cfg.max_counter
        logger.debug(f"Heartbeat Counter: {heartbeat_counter}")


class Heartbeat:
    def __init__(self, cfg: HeartbeatConfig, coreio_origin: str) -> None:
        self.cfg = cfg
        self.coreio_origin = coreio_origin
        self._heartbeat_thread: Thread = self.start_heartbeat(cfg, coreio_origin)

    def start_heartbeat(self, cfg: HeartbeatConfig, coreio_origin: str) -> Thread:
        heartbeat_thread = threading.Thread(
            target=heartbeat,
            args=(
                cfg,
                coreio_origin,
            ),
            daemon=True,
            name="corerl_heartbeat",
        )
        heartbeat_thread.start()
        return heartbeat_thread

    def healthcheck(self):
        if self.cfg.heartbeat_node_id is not None and not self._heartbeat_thread.is_alive():
            logger.error("Heartbeat stopped -- defibrillating...")
            self.start_heartbeat(self.cfg, self.coreio_origin)
