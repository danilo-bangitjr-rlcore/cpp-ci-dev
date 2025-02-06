import asyncio
import logging
import threading
from datetime import timedelta
from threading import Thread

from asyncua import Client, ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256

from corerl.configs.config import config
from corerl.environment.async_env.async_env import OPCEnvConfig
from corerl.utils.time import clock_generator, wait_for_timestamp

logger = logging.getLogger(__name__)


@config()
class HeartbeatConfig:
    heartbeat_node_id: str | None = None
    heartbeat_period: timedelta = timedelta(seconds=5)
    max_counter: int = 1000


async def heartbeat(cfg: HeartbeatConfig, opc_env_config: OPCEnvConfig):
    if cfg.heartbeat_node_id is None:
        return

    opc_client = Client(opc_env_config.opc_conn_url)

    if opc_env_config.client_cert_path and opc_env_config.client_private_key_path:
        # NOTE: this does not exist within the Sync variant of OPC Client and is the source of why we need to
        # add these hacky async snippets into our synchronous codebase
        await opc_client.set_security(
            SecurityPolicyBasic256Sha256,
            certificate=opc_env_config.client_cert_path,
            private_key=opc_env_config.client_private_key_path,
            mode=ua.MessageSecurityMode.SignAndEncrypt,
            server_certificate=opc_env_config.server_cert_path,
        )
    async with opc_client:
        heartbeat_node = opc_client.get_node(cfg.heartbeat_node_id)

    heartbeat_clock = clock_generator(tick_period=cfg.heartbeat_period)
    heartbeat_counter = 0

    while True:
        # write counter
        async with opc_client:
            await opc_client.write_values([heartbeat_node], [float(heartbeat_counter)])

        # increment counter
        heartbeat_counter += 1
        heartbeat_counter %= cfg.max_counter
        logger.debug(f"{heartbeat_counter=}")

        # wait for next heartbeat
        next_heartbeat_ts = next(heartbeat_clock)
        wait_for_timestamp(next_heartbeat_ts)


class Heartbeat:
    def __init__(self, cfg: HeartbeatConfig, opc_env_config: OPCEnvConfig) -> None:
        self.cfg = cfg
        self.opc_env_config = opc_env_config
        self._heartbeat_thread: Thread = self.start_heartbeat(cfg, opc_env_config)

    def start_heartbeat(self, cfg: HeartbeatConfig, opc_env_config: OPCEnvConfig) -> Thread:
        heartbeat_thread = threading.Thread(
            target=asyncio.run,
            args=(heartbeat(cfg, opc_env_config),),
            daemon=True,
            name="corerl_heartbeat",
        )
        heartbeat_thread.start()
        return heartbeat_thread

    def healthcheck(self):
        if self.cfg.heartbeat_node_id is not None and not self._heartbeat_thread.is_alive():
            logger.error("Heartbeat stopped -- defibrillating...")
            self.start_heartbeat(self.cfg, self.opc_env_config)
