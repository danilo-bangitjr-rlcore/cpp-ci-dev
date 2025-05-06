from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from corerl.configs.config import MISSING, computed, config
from corerl.messages.heartbeat import HeartbeatConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


# -------------
# -- Configs --
# -------------
@config()
class InteractionConfig:
    name: Literal["sim_interaction", "dep_interaction"] = "dep_interaction"

    obs_period: timedelta = MISSING
    """
    Kind: required external

    The duration between observations. Sensor readings received within this
    interval will be aggregated together to produce a single reading.
    """

    action_period: timedelta = MISSING
    """
    Kind: required external

    The expected duration between AI-controlled setpoint changes. Should
    be a multiple of obs_period.
    """

    state_age_tol: timedelta = MISSING
    """
    Kind: internal

    How long until interaction state is stale
    """

    update_period: timedelta = MISSING
    """
    Kind: internal

    The duration between agent updates. Syncronized with obs_period
    by default.
    """

    setpoint_ping_period: timedelta | None = None
    """
    Kind: internal

    How often to ping setpoints.
    """

    load_historical_data: bool = True
    """
    Kind: internal

    Whether or not to load historical data.
    """

    update_warmup: int = 0 # number of updates before interacting
    """
    Kind: internal

    The number of updates to apply before the first interaction.
    """

    historical_batch_size: int = 10000
    """
    Kind: internal

    The number of historical observations to request from TSDB
    on every obs_period. Primarily controls the computational
    impact of loading historical data.
    """

    hist_chunk_start: datetime | None = None
    """
    Kind: optional external

    The start time of the first historical chunk to fetch.
    """

    checkpoint_path: Path = MISSING
    """
    Kind: internal

    The path to the checkpoint directory.
    """

    restore_checkpoint: bool = True
    """
    Kind: internal

    Whether to restore the agent from the latest checkpoint.
    """

    warmup_period: timedelta | None = None
    """
    Kind: internal

    The number of datapoints to preload into the pipeline
    before considering transitions valid. Used to warmup
    history-tracking features, such as state traces.
    """

    checkpoint_freq: timedelta = Field(default_factory=lambda: timedelta(hours=1))
    """
    Kind: internal

    How frequently we perform checkpointing.
    """

    checkpoint_cliff: timedelta = Field(default_factory=lambda: timedelta(hours=24))
    """
    Kind: internal

    Specifies period in time where we keep all checkpoints, before beginning to delete them.
    """

    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


    @computed('update_period')
    @classmethod
    def _update_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period

    @computed('state_age_tol')
    @classmethod
    def _state_age_tol(cls, cfg: MainConfig):
        return cfg.interaction.obs_period

    @computed('checkpoint_path')
    @classmethod
    def _checkpoint_path(cls, cfg: 'MainConfig'):
        return Path('outputs') / cfg.agent_name / (f'seed-{cfg.seed}') / 'checkpoints'
