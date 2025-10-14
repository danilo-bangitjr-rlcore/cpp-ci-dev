from datetime import timedelta

from lib_config.config import config


@config()
class HeartbeatConfig:
    connection_id: str | None = None
    heartbeat_node_id: str | None = None
    heartbeat_period: timedelta = timedelta(seconds=5)
    max_counter: int = 1000
