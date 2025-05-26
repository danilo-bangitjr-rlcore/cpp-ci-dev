from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from corerl.configs.config import MISSING, computed, config, list_, post_processor
from corerl.messages.heartbeat import HeartbeatConfig
from corerl.utils.maybe import Maybe

if TYPE_CHECKING:
    from corerl.config import MainConfig


# -------------
# -- Configs --
# -------------
@config()
class InteractionConfig:
    name: Literal["sim_interaction", "dep_interaction"] = "dep_interaction"

    time_dilation: float = 1.0
    """
    Kind: internal

    Time dilation factor. This number will divider all time-based configs to enable
    faster-than-life running. E.g. new_obs_period = obs_period / time_dilation
    """

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

    load_historical_data: bool = MISSING
    """
    Kind: internal

    Whether or not to load historical data.
    """

    historical_windows: list[tuple[datetime | None, datetime | None]] = list_([(None, None)])
    """
    Kind: optional external

    Time windows specified by tuples of (start_time, end_time) from which to load historical data.
    If either the start or end time are None, the default behavior is to replace the None with the
    earliest or latest recorded timestamp, respectively. Nones may only appear as the start_time of
    the first window or the end_time of the last window.
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

    write_obs_to_csv: bool = False
    """
    Kind: internal

    Whether to write observations to a csv file
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

    @computed('load_historical_data')
    @classmethod
    def _load_historical_data(cls, cfg: MainConfig):
        if cfg.interaction.name == "sim_interaction":
            return False
        else:
            return True

    @post_processor
    def _validate_hist_windows(self, cfg: MainConfig):
        for i, (start, stop) in enumerate(self.historical_windows):
            start = Maybe(start)
            stop = Maybe(stop)

            # ensure Nones only appear at beginning or end of the sequence
            assert i == 0 or start.is_some()
            assert i == len(self.historical_windows) - 1 or stop.is_some()

            # if there is a None, don't check consistency
            if start.is_none() or stop.is_none():
                continue

            assert start.expect() < stop.expect()
