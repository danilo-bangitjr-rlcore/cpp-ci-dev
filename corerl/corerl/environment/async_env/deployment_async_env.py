import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import TypedDict

import numpy as np
import pandas as pd
from coreio.utils.io_events import OPCUANodeWriteValue
from lib_defs.config_defs.tag_config import TagType
from lib_utils.iterable import partition
from lib_utils.maybe import Maybe

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.environment.async_env.async_env import AsyncEnv, AsyncEnvConfig
from corerl.tags.setpoint import SetpointTagConfig, eval_bound
from corerl.tags.tag_config import TagConfig, get_scada_tags
from corerl.utils.coreio import CoreIOLink

logger = logging.getLogger(__name__)


class ActionNodeData(TypedDict):
    connection_id: str
    node_id: str


class DeploymentAsyncEnv(AsyncEnv):
    def __init__(self, cfg: AsyncEnvConfig, tag_configs: Sequence[TagConfig]):
        self._cfg = cfg
        self.tag_configs = tag_configs
        self.obs_period = cfg.obs_period
        self.tag_names = [
            tag_cfg.name
            for tag_cfg in get_scada_tags(tag_configs)
        ]
        self.tag_aggs = {
            tag_cfg.name: tag_cfg.agg
            for tag_cfg in get_scada_tags(tag_configs)
        }
        self._action_cfgs = {
            tag_cfg.name: tag_cfg
            for tag_cfg in tag_configs
            if tag_cfg.type == TagType.ai_setpoint
        }

        self.action_nodes = self._build_action_nodes(self._action_cfgs)
        self.coreio_client = self._init_thinclient()
        self.data_reader = self._init_datareader()

    # ------------------
    # -- Initializers --
    # ------------------
    def _build_action_nodes(self, action_cfgs: dict[str, SetpointTagConfig]):
        def _build_action_node_entry(tag_cfg: SetpointTagConfig) -> ActionNodeData:
            assert tag_cfg.node_identifier is not None, "Tag Config action missing node_identifier"
            assert tag_cfg.connection_id is not None, "Tag Config action missing connection_id"
            logger.info(f"Mapping ai_setpoint '{tag_cfg.name}' -> OPC node id '{tag_cfg.node_identifier}' on conn '{tag_cfg.connection_id}'") # noqa: E501
            return ActionNodeData(
                connection_id=tag_cfg.connection_id, node_id=tag_cfg.node_identifier,
            )

        sorted_cfgs = sorted(action_cfgs.values(), key=lambda cfg: cfg.name)
        return {
            tag_cfg.name: _build_action_node_entry(tag_cfg)
            for tag_cfg in sorted_cfgs
        }

    def _init_thinclient(self):
        return CoreIOLink(self._cfg.coreio_origin)

    def _init_datareader(self):
        return DataReader(db_cfg=self._cfg.db)


    # ----------------
    # -- Public API --
    # ----------------
    def close(self):
        self.data_reader.close()

    def emit_action(self, action: pd.DataFrame, log_action: bool = False) -> None:
        """Writes directly to the OPC server"""
        sanitize_actions(action, self._action_cfgs)

        if log_action:
            logger.info("--- Emitting action ---")
            for line in action.to_string().splitlines():
                logger.info(line)

        def _build_payload(action_name: str) -> tuple[str, OPCUANodeWriteValue]:
            conn_id = self.action_nodes[action_name]["connection_id"]
            node_id = self.action_nodes[action_name]["node_id"]
            action_val = action[action_name].iloc[0].item()
            return (conn_id, OPCUANodeWriteValue(node_id=node_id, value=action_val))

        write_payloads = partition(
            _build_payload(action_name) for action_name in action.columns
        )

        try:
            self.coreio_client.write_opcua_nodes(write_payloads)
        except Exception:
            logger.exception("emit_action failed to write to coreio")


    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        return self.data_reader.single_aggregated_read(
            names=self.tag_names, start_time=now - self.obs_period, end_time=now, tag_aggregations=self.tag_aggs,
        )


    def get_cfg(self):
        return self._cfg


# ---------------
# -- Utilities --
# ---------------
def sanitize_actions(action: pd.DataFrame, action_cfgs: Mapping[str, SetpointTagConfig], rtol: float = 0.001) -> None:
    if len(action) < 1:
        logger.error("Action df empty")
        return
    if len(action) > 1:
        logger.error(f"Action df contains {len(action)} rows, clearing action df")
        action.drop(columns=action.columns, inplace=True)
        return

    clip_action(action, action_cfgs, rtol)

def clip_action(action: pd.DataFrame, action_cfgs: Mapping[str, SetpointTagConfig], rtol: float = 0.001) -> None:
    for action_name, action_cfg in action_cfgs.items():
        action_val = action[action_name].iloc[0]
        lo, hi = get_clip_bounds(action_cfg, action)
        atol = rtol * (hi - lo)

        if (action_val < lo) or (action_val > hi):
            logger.error(
                f"Action {action_cfg.name} assigned value {action_val}, outside of operating range [{lo}, {hi}]",
            )

        action[action_name] = np.clip(action_val, lo + atol, hi - atol)

def get_clip_bounds(action_cfg: SetpointTagConfig, action: pd.DataFrame):
    # prefer to use red zones, otherwise use operating range
    lo = (
        Maybe[float | str](action_cfg.red_bounds and action_cfg.red_bounds[0])
        .map(partial(eval_bound, action, "lo", action_cfg.red_bounds_func, action_cfg.red_bounds_tags))
        .otherwise(lambda: action_cfg.operating_range and action_cfg.operating_range[0])
    ).expect()

    hi = (
        Maybe[float | str](action_cfg.red_bounds and action_cfg.red_bounds[1])
        .map(partial(eval_bound, action, "hi", action_cfg.red_bounds_func, action_cfg.red_bounds_tags))
        .otherwise(lambda: action_cfg.operating_range and action_cfg.operating_range[1])
    ).expect()
    return lo, hi
