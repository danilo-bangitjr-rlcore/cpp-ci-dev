import logging
from datetime import UTC, datetime
from functools import partial
from typing import TypedDict

import numpy as np
import pandas as pd

# Data Pipline
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import FloatBounds, TagConfig, eval_bound
from corerl.environment.async_env.async_env import AsyncEnv, DepAsyncEnvConfig
from corerl.utils.coreio import CoreIOThinClient, OPCUADataType, OPCUANodeWriteValue
from corerl.utils.maybe import Maybe

logger = logging.getLogger(__name__)


class ActionNodeData(TypedDict):
    connection_id: str
    node_id: str
    data_type: OPCUADataType


class DeploymentAsyncEnv(AsyncEnv):
    """AsyncEnv which communicates actions through OPC and retrieves observations through TSDB.
    Ensure that TimescaleDB, Telegraf, our OPC Server, and our simulated OPC environment is running prior to use.
    """

    def __init__(self, cfg: DepAsyncEnvConfig, tag_configs: list[TagConfig]):
        self.cfg = cfg

        self.coreio_client = CoreIOThinClient(cfg.coreio_origin)

        self.tag_configs = tag_configs
        self.obs_period = cfg.obs_period
        self.action_period = cfg.action_period
        self.action_tolerance = cfg.action_tolerance

        self.tag_names = [tag.name for tag in tag_configs]
        self.tag_aggs = {tag.name: tag.agg for tag in tag_configs}
        self._meta_tags = [tag for tag in tag_configs if tag.is_meta]
        self._observation_tags = [tag for tag in tag_configs if not tag.is_meta and tag.action_constructor is None]

        self.data_reader = DataReader(db_cfg=cfg.db)

        # create dict of action tags
        action_cfgs = [tag for tag in tag_configs if tag.action_constructor is not None]
        self._action_cfgs: dict[str, TagConfig] = {}
        for tag_cfg in sorted(action_cfgs, key=lambda cfg: cfg.name):
            self._action_cfgs[tag_cfg.name] = tag_cfg

        # define opc action nodes
        self.action_nodes: dict[str, ActionNodeData] = {}

        for tag_cfg in sorted(tag_configs, key=lambda cfg: cfg.name):
            if tag_cfg.action_constructor is None:
                continue

            tag_name = tag_cfg.name
            assert tag_cfg.node_identifier is not None, "Tag Config action missing node_identifier"
            node_id = tag_cfg.node_identifier
            assert tag_cfg.connection_id is not None, "Tag Config action missing connection_id"
            connection_id = tag_cfg.connection_id

            logger.info(f"Registering action '{tag_name}' with OPC node id '{node_id}' on conn '{connection_id}'")
            resp_payload = self.coreio_client.read_opcua_node(connection_id, node_id)
            raw_data_type = resp_payload["value"]["dataType"]
            data_type: OPCUADataType | int | None = None
            if isinstance(raw_data_type, str):
                data_type = OPCUADataType[raw_data_type]
            elif isinstance(raw_data_type, int):
                data_type = OPCUADataType(raw_data_type)
            assert data_type is not None, "Tag Config action failed to determine data_type"
            self.action_nodes[tag_name] = ActionNodeData(
                connection_id=connection_id, node_id=node_id, data_type=data_type
            )

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

        # if action df got nuked in sanitizer, this for loop does nothing
        write_payloads: dict[str, list[OPCUANodeWriteValue]] = {}
        for action_name in action.columns:
            connection_id = self.action_nodes[action_name].get("connection_id")
            node_id = self.action_nodes[action_name].get("node_id")
            data_type = self.action_nodes[action_name].get("data_type")
            if data_type in {
                OPCUADataType.SByte,
                OPCUADataType.Byte,
                OPCUADataType.Int16,
                OPCUADataType.UInt16,
                OPCUADataType.Int32,
                OPCUADataType.UInt32,
                OPCUADataType.Int64,
                OPCUADataType.UInt64,
            }:
                action_val = int(action[action_name].iloc[0])
            elif data_type in {OPCUADataType.Double, OPCUADataType.Float}:
                action_val = float(action[action_name].iloc[0])
            else:
                logger.warning("Action OPC dtype undefined")
                action_val = action[action_name].iloc[0]

            if connection_id not in write_payloads:
                write_payloads[connection_id] = []

            write_payloads[connection_id].append(OPCUANodeWriteValue(node_id, action_val, data_type))

        for k, v in write_payloads.items():
            try:
                self.coreio_client.write_opcua_nodes(k, v)
            except Exception:
                logger.exception("emit_action failed to write to coreio")

    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        obs = self.data_reader.single_aggregated_read(
            names=self.tag_names, start_time=now - self.obs_period, end_time=now, tag_aggregations=self.tag_aggs
        )
        return obs

    def maybe_write_agent_step(self, step: int):
        # removed within https://github.com/rlcoretech/core-rl/commit/fff373f4018a7a3d9ffee36868f354b89a9ee640
        # also currently incompatible with thin client
        return step

    def get_cfg(self):
        return self.cfg


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
        lo, hi = get_clip_bounds(action_cfg, action)

        if (action_val < lo) or (action_val > hi):
            logger.error(
                f"Action {action_cfg.name} assigned value {action_val}, outside of operating range [{lo}, {hi}]"
            )

        action[action_name] = np.clip(action_val, lo, hi)

def get_clip_bounds(action_cfg: TagConfig, action: pd.DataFrame) -> FloatBounds:
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
