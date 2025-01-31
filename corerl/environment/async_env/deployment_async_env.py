import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from asyncua.sync import Client, SyncNode
from asyncua.ua.uaerrors import BadNodeIdUnknown

from corerl.configs.config import MISSING, config

# Data Pipline
from corerl.data_pipeline.bound_checker import Bounds
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, OPCEnvConfig, TSDBEnvConfig
from corerl.utils.maybe import Maybe
from corerl.utils.opc_connection import make_opc_node_id

logger = logging.getLogger(__name__)

@config()
class DepAsyncEnvConfig(TSDBEnvConfig, OPCEnvConfig):
    name: str = "dep_async_env"
    action_tolerance: timedelta = MISSING


class DeploymentAsyncEnv(AsyncEnv):
    """AsyncEnv which communicates actions through OPC and retrieves observations through TSDB.
    Ensure that TimescaleDB, Telegraf, our OPC Server, and our simulated OPC environment is running prior to use.
    """

    def __init__(self, cfg: DepAsyncEnvConfig, tag_configs: list[TagConfig]):
        self.url = cfg.opc_conn_url
        self.ns = cfg.opc_ns

        self.tag_configs = tag_configs
        self.obs_period = cfg.obs_period
        self.action_period = cfg.action_period
        self.action_tolerance = cfg.action_tolerance

        self.tag_names = [tag.name for tag in tag_configs]
        self._meta_tags = [tag for tag in tag_configs if tag.is_meta]
        self._observation_tags = [
            tag for tag in tag_configs
            if not tag.is_meta and tag.action_constructor is None
        ]

        self.data_reader = DataReader(db_cfg=cfg.db)

        # create dict of action tags
        action_cfgs = [tag for tag in tag_configs if tag.action_constructor is not None]
        self._action_cfgs: dict[str, TagConfig] = {}
        for tag_cfg in sorted(action_cfgs, key=lambda cfg: cfg.name):
            self._action_cfgs[tag_cfg.name] = tag_cfg

        # define opc action nodes
        self.action_nodes: dict[str, SyncNode] = {}
        with Client(self.url) as opc_client:
            for tag_cfg in sorted(tag_configs, key=lambda cfg: cfg.name):
                if tag_cfg.action_constructor is None:
                    continue

                tag_name = tag_cfg.name
                if tag_cfg.node_identifier is not None:
                    node_name = tag_cfg.node_identifier
                else:
                    node_name = tag_name

                id = make_opc_node_id(node_name, cfg.opc_ns)
                node = opc_client.get_node(id)
                logger.info(f"Registering action '{tag_name}' with OPC node id '{id}'")
                self.action_nodes[tag_name] = node

            try:
                id = make_opc_node_id("agent_step", self.ns)
                self.agent_step_node = opc_client.get_node(id)
            except BadNodeIdUnknown:
                self.agent_step_node = None

    def _make_opc_node_id(self, str_id: str, namespace: int = 0):
        return f"ns={namespace};s={str_id}"

    def close(self):
        """Closes the opc client and data reader
        Can also use __exit__ or cleanup
        """
        self.data_reader.close()

    def emit_action(self, action: pd.DataFrame, log_action: bool = False) -> None:
        """Writes directly to the OPC server"""
        sanitize_actions(action, self._action_cfgs)

        if log_action:
            logger.info("--- Emitting action ---")
            [logger.info(line) for line in action.to_string().splitlines()]
        with Client(self.url) as opc_client:
            # if action df got nuked in sanitizer, this for loop does nothing
            for action_name in action.columns:
                node = self.action_nodes[action_name]
                action_val = float(action[action_name].iloc[0])
                opc_client.write_values([node], [action_val])


    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        obs = self.data_reader.single_aggregated_read(
            names=self.tag_names, start_time=now - self.obs_period, end_time=now
        )
        return obs

    def maybe_write_agent_step(self, step: int):
        if self.agent_step_node is None:
            return

        with Client(self.url) as opc_client:
            logger.info(f"Incrementing agent step node to {step}")
            opc_client.write_values([self.agent_step_node], [float(step)])


def sanitize_actions(action: pd.DataFrame, action_cfgs: dict[str, TagConfig]) -> None:
    if len(action) < 1:
        logger.error("Action df empty")
        return
    if len(action) > 1:
        logger.error(f"Action df contains {len(action)} rows, clearing action df")
        action.drop(columns=action.columns, inplace=True)
        return

    clip_action(action, action_cfgs)

def clip_action(action: pd.DataFrame, action_cfgs: dict[str, TagConfig]) -> None:
    for action_name in action_cfgs.keys():
        action_cfg = action_cfgs[action_name]
        action_val = action[action_name].iloc[0]
        lo, hi = get_clip_bounds(action_cfg)

        if (action_val < lo) or (action_val > hi):
            logger.error(
                f"Action {action_cfg.name} assigned value {action_val}, outside of operating range [{lo}, {hi}]"
            )

        action[action_name] = np.clip(action_val, lo, hi)

def get_clip_bounds(action_cfg: TagConfig) -> Bounds:
    # prefer to use red zones, otherwise use operating range
    lo = (
        Maybe[float](action_cfg.red_bounds and action_cfg.red_bounds[0])
        .otherwise(lambda: action_cfg.operating_range and action_cfg.operating_range[0])
    ).expect()

    hi = (
        Maybe[float](action_cfg.red_bounds and action_cfg.red_bounds[1])
        .otherwise(lambda: action_cfg.operating_range and action_cfg.operating_range[1])
    ).expect()
    return lo, hi
