import logging
import threading
from datetime import timedelta
from threading import Thread

from asyncua.sync import Client

from corerl.configs.config import config
from corerl.utils.time import clock_generator, wait_for_timestamp

logger = logging.getLogger(__name__)


@config()
class HeartbeatConfig:
    opc_conn_url: str | None = None
    heartbeat_node_id: str | None = None
    heartbeat_period: timedelta = timedelta(seconds=5)
    max_counter: int = 1000


def heartbeat(cfg: HeartbeatConfig):
    if cfg.opc_conn_url is None or cfg.heartbeat_node_id is None:
        return

    with Client(cfg.opc_conn_url) as opc_client:
        heartbeat_node = opc_client.get_node(cfg.heartbeat_node_id)

    heartbeat_clock = clock_generator(tick_period=cfg.heartbeat_period)
    heartbeat_counter = 0

    while True:
        # write counter
        with Client(cfg.opc_conn_url) as opc_client:
            opc_client.write_values([heartbeat_node], [float(heartbeat_counter)])

        # increment counter
        heartbeat_counter += 1
        heartbeat_counter %= cfg.max_counter
        logger.debug(f"{heartbeat_counter=}")

        # wait for next heartbeat
        next_heartbeat_ts = next(heartbeat_clock)
        wait_for_timestamp(next_heartbeat_ts)


class Heartbeat:
    def __init__(self, cfg: HeartbeatConfig) -> None:
        self.cfg = cfg
        self._heartbeat_thread: Thread = self.start_heartbeat(cfg)

    def start_heartbeat(self, cfg: HeartbeatConfig) -> Thread:
        heartbeat_thread = threading.Thread(
            target=heartbeat,
            args=(cfg,),
            daemon=True,
            name="corerl_heartbeat",
        )
        heartbeat_thread.start()
        return heartbeat_thread

    def healthcheck(self):
        if self.cfg.opc_conn_url is not None and not self._heartbeat_thread.is_alive():
            logger.error("Heartbeat stopped -- defibrillating...")
            self.start_heartbeat(self.cfg)
