import asyncio
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from asyncua import Client, Node, ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from pydantic import BaseModel, ConfigDict

# Data Pipline
from corerl.data_pipeline.bound_checker import Bounds
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, DepAsyncEnvConfig
from corerl.utils.maybe import Maybe
from corerl.utils.opc_connection import make_opc_node_id

logger = logging.getLogger(__name__)

class SyncNodeData(BaseModel):
    node: Node
    var_type: ua.VariantType
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DeploymentAsyncEnv(AsyncEnv):
    """AsyncEnv which communicates actions through OPC and retrieves observations through TSDB.
    Ensure that TimescaleDB, Telegraf, our OPC Server, and our simulated OPC environment is running prior to use.
    """

    def __init__(self, cfg: DepAsyncEnvConfig, tag_configs: list[TagConfig]):
        self.cfg = cfg

        self.url = cfg.opc_conn_url
        self.ns = cfg.opc_ns
        self.client_cert_path = cfg.client_cert_path
        self.client_private_key_path = cfg.client_private_key_path
        self.server_cert_path = cfg.server_cert_path
        self.application_uri = cfg.application_uri

        self.tag_configs = tag_configs
        self.obs_period = cfg.obs_period
        self.action_period = cfg.action_period
        self.action_tolerance = cfg.action_tolerance

        self.tag_names = [tag.name for tag in tag_configs]
        self.tag_aggs = {tag.name: tag.agg for tag in tag_configs}
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
        self.action_nodes: dict[str, SyncNodeData] = {}
        self.agent_step_node: Node | None = None

        # hacky initialization of OPC client with security settings
        async def _init_opc_client(cfg: DepAsyncEnvConfig, tag_configs: list[TagConfig]):
            """TODO: remove these opc_client workarounds once OPC logic is pulled from corerl
            """
            opc_client = Client(self.url)

            if cfg.client_cert_path and cfg.client_private_key_path:
                assert self.application_uri is not None
                opc_client.application_uri = self.application_uri
                # NOTE: this does not exist within the Sync variant of OPC Client and is the source of why we need to
                # add these hacky async snippets into our synchronous codebase
                await opc_client.set_security(
                    SecurityPolicyBasic256Sha256,
                    certificate=cfg.client_cert_path,
                    private_key=cfg.client_private_key_path,
                    mode=ua.MessageSecurityMode.SignAndEncrypt,
                    server_certificate=cfg.server_cert_path,
                )

            async with opc_client:
                for tag_cfg in sorted(tag_configs, key=lambda cfg: cfg.name):
                    if tag_cfg.action_constructor is None:
                        continue

                    tag_name = tag_cfg.name
                    if tag_cfg.node_identifier is None:
                        node_name = tag_name
                        id = make_opc_node_id(node_name, cfg.opc_ns)
                    else:
                        # PR 531: assume node_identifier is the full OPC node identifier, fallback to just identifier
                        # and construct the full node id using ns defined within cfg.opc_ns
                        if isinstance(tag_cfg.node_identifier, str) and tag_cfg.node_identifier.startswith("ns="):
                            id = tag_cfg.node_identifier
                        else:
                            id = make_opc_node_id(tag_cfg.node_identifier, cfg.opc_ns)
                            logger.warning(f"node_identifier defined without ns: {tag_cfg.node_identifier}")

                    node = opc_client.get_node(id)
                    var_type = await node.read_data_type_as_variant_type()
                    logger.info(f"Registering action '{tag_name}' with OPC node id '{id}'")
                    self.action_nodes[tag_name] = SyncNodeData(node=node, var_type=var_type)

        asyncio.run(_init_opc_client(cfg, tag_configs))


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

        async def _async_opc_emit_action(action: pd.DataFrame):
            """TODO: remove these opc_client workarounds once OPC logic is pulled from corerl
            """
            async with Client(self.url) as opc_client:
                # if action df got nuked in sanitizer, this for loop does nothing
                for action_name in action.columns:
                    node = self.action_nodes[action_name].node
                    var_type = self.action_nodes[action_name].var_type
                    action_val = float(action[action_name].iloc[0])
                    # the source timestamp is sent to the OPC server, which itself has a server timestamp
                    # recorded when it receives the write. if these values are too far apart, some OPC
                    # implementations will consider the quality of this tag to be bad, so we need
                    # to ensure that the values we write have an up-to-date timestamp
                    # (and that they align with the server).
                    dt = ua.uatypes.DateTime.now(UTC) # this is a load bearing timestamp
                    data_value = ua.DataValue(ua.Variant(action_val, var_type), SourceTimestamp=dt)
                    await opc_client.write_values([node], [data_value])

        asyncio.run(_async_opc_emit_action(action))


    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        obs = self.data_reader.single_aggregated_read(
            names=self.tag_names, start_time=now - self.obs_period, end_time=now, tag_aggregations=self.tag_aggs
        )
        return obs

    def maybe_write_agent_step(self, step: int):
        if self.agent_step_node is None:
            return

        async def _async_opc_write_agent_step(step_node: Node):
            async with Client(self.url) as opc_client:
                logger.info(f"Incrementing agent step node to {step}")
                await opc_client.write_values([step_node], [float(step)])

        asyncio.run(_async_opc_write_agent_step(self.agent_step_node))

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
