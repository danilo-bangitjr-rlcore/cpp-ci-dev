from typing import TYPE_CHECKING, Literal

from lib_agent.critic.critic_utils import RollingResetConfig
from lib_agent.gamma_schedule import GammaScheduleConfig
from lib_config.config import MISSING, computed, config
from pydantic import Field

from corerl.configs.agent.base import BaseAgentConfig
from corerl.configs.agent.buffer_configs import MixedHistoryBufferConfig, RecencyBiasBufferConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

BufferConfig = MixedHistoryBufferConfig | RecencyBiasBufferConfig


@config()
class CriticNetworkConfig:
    ensemble: int = MISSING

    @computed('ensemble')
    @classmethod
    def _ensemble(cls, cfg: 'MainConfig'):
        return cfg.feature_flags.ensemble


@config()
class GTDCriticConfig:
    polyak_tau: float = 0.0
    action_regularization: float = 0.0
    action_regularization_epsilon: float = 0.1
    buffer: BufferConfig = MISSING
    all_layer_norm: bool = False
    stepsize: float = 0.0001
    critic_network: CriticNetworkConfig = Field(default_factory=CriticNetworkConfig)
    rolling_reset_config: RollingResetConfig = Field(default_factory=RollingResetConfig)

    @computed('buffer')
    @classmethod
    def _buffer(cls, cfg: 'MainConfig'):
        if cfg.feature_flags.recency_bias_buffer:
            return RecencyBiasBufferConfig(
                obs_period=int(cfg.interaction.obs_period.total_seconds()),
                gamma=[cfg.agent.gamma],
                effective_episodes=[100],
                ensemble=1,
                ensemble_probability=1.0,
                id='actor',
            )
        buffer_cfg = MixedHistoryBufferConfig(id='actor')
        buffer_cfg.ensemble = 1
        buffer_cfg.ensemble_probability = 1

        return buffer_cfg


@config()
class AdvCriticConfig(GTDCriticConfig):
    """Config for Advantage-based critic."""
    num_policy_actions: int = 100
    advantage_centering_weight: float = 0.1
    adv_l2_regularization: float = 1.0
    h_lr_mult: float = 1.0
    v_lr_mult: float = 1.0


@config()
class PercentileActorConfig:
    num_samples: int = 128
    actor_percentile: float = 0.05
    proposal_percentile: float = 0.2
    prop_percentile_learned: float = 0.8
    sort_noise: float = 0.0
    actor_stepsize: float = 0.0001
    sampler_stepsize: float = 0.0001
    mu_multiplier: float = 1.0
    sigma_multiplier: float = 1.0
    ensemble_aggregation: Literal["mean", "percentile", "ucb"] = "mean"
    ensemble_percentile: float = 0.5
    sigma_regularization: float = 0.0
    buffer: BufferConfig = MISSING
    even_better_q: bool = False
    std_bonus: float = 1.0

    @computed('buffer')
    @classmethod
    def _buffer(cls, cfg: 'MainConfig'):
        if cfg.feature_flags.recency_bias_buffer:
            buffer_cfg = RecencyBiasBufferConfig(
                obs_period=int(cfg.interaction.obs_period.total_seconds()),
                gamma=[cfg.agent.gamma],
                effective_episodes=[100],
                ensemble=1,
                ensemble_probability=1.0,
                id='actor',
            )
        else:
            buffer_cfg = MixedHistoryBufferConfig(id='actor')
            buffer_cfg.ensemble = 1
            buffer_cfg.ensemble_probability = 1

        return buffer_cfg


@config()
class GreedyACConfig(BaseAgentConfig):
    """
    Kind: internal

    Agent hyperparameters. For internal use only.
    These should never be modified for production unless
    for debugging. These may be modified in tests and
    research to illicit particular behaviors.
    """
    name: Literal["greedy_ac", "gaac"] = "greedy_ac"

    critic: GTDCriticConfig | AdvCriticConfig = Field(default_factory=GTDCriticConfig)
    policy: PercentileActorConfig = Field(default_factory=PercentileActorConfig)
    gamma_schedule: GammaScheduleConfig = Field(
        default_factory=lambda: GammaScheduleConfig(type='identity'),
    )

    loss_threshold: float = 0.0001
    """
    Kind: internal

    Minimum desired change in loss between updates. If the loss value changes
    by more than this magnitude, then continue performing updates.
    """

    weight_decay: float = 0.001
    """
    Kind: internal

    Weight decay parameter for adam optimizer
    """

    loss_ema_factor: float = 0.75
    """
    Kind: internal

    Exponential moving average factor for early stopping based on loss.
    Closer to 1 means slower update to avg, closer to 0 means less averaging.
    """

    max_internal_actor_updates: int = 3
    """
    Number of actor updates per critic update. Early stopping is done
    using the loss_threshold. A minimum of 1 update will always be performed.
    """

    max_critic_updates: int = 10
    """
    Number of critic updates. Early stopping is done using the loss_threshold.
    A minimum of 1 update will always be performed.
    """

    bootstrap_action_samples: int = 10
    """
    Number of action samples to use for bootstrapping,
    producing an Expected Sarsa-like update.
    """

    max_action_stddev: float = 3
    """
    Maximum number of stddevs from the mean for the action
    taken during an interaction step. Forcefully prevents
    very long-tailed events from occurring.
    """

    return_scale: float = 1.0
    """
    Returns predicted by the critic are scaled by this value.
    """

    @computed('gamma_schedule')
    @classmethod
    def _gamma_schedule(cls, cfg: 'MainConfig'):
        if cfg.feature_flags.gamma_schedule:
            error_msg = "gamma schedule is only supported for n=1"
            assert cfg.pipeline.trajectory_creator.min_n_step == 1, error_msg
            assert cfg.pipeline.trajectory_creator.max_n_step == 1, error_msg
            assert cfg.interaction.obs_period == cfg.interaction.action_period, error_msg

            if cfg.max_steps is not None:
                horizon = min(cfg.max_steps, 2)
            else:
                horizon = cfg.agent.gamma_schedule.horizon
            return GammaScheduleConfig(
                    type='logarithmic',
                    max_gamma=cfg.agent.gamma,
                    update_interval=cfg.agent.gamma_schedule.update_interval,
                    horizon=horizon,
                )
        return None
