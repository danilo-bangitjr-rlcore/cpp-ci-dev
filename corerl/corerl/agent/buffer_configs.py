from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Literal

from lib_agent.buffer.mixed_history_buffer import MixedHistoryBufferConfig as LibMixedHistoryBufferConfig
from lib_agent.buffer.recency_bias_buffer import RecencyBiasBufferConfig as LibRecencyBiasBufferConfig
from lib_config.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class RecencyBiasBufferConfig:
    name: Literal["recency_bias_buffer"] = "recency_bias_buffer"
    obs_period: int = MISSING
    gamma: Sequence[float] = field(default_factory=lambda: [0.99])
    effective_episodes: Sequence[int] = field(default_factory=lambda: [100])
    ensemble: int = MISSING
    uniform_weight: float = 0.01
    ensemble_probability: float = 0.5
    batch_size: int = 32
    max_size: int = 1_000_000
    seed: int = 0
    n_most_recent: int = 1
    id: str = ""

    @computed('ensemble')
    @classmethod
    def _ensemble(cls, cfg: "MainConfig"):
        return cfg.feature_flags.ensemble + cfg.agent.critic.rolling_reset_config.num_background_critics

    @computed('gamma')
    @classmethod
    def _gamma(cls, cfg: 'MainConfig'):
        return cfg.agent.gamma

    @computed('obs_period')
    @classmethod
    def _obs_period(cls, cfg: 'MainConfig'):
        return cfg.interaction.obs_period

    def to_lib_config(self) -> LibRecencyBiasBufferConfig:
        return LibRecencyBiasBufferConfig(
            name=self.name,
            obs_period=self.obs_period,
            gamma=self.gamma,
            effective_episodes=self.effective_episodes,
            ensemble=self.ensemble,
            uniform_weight=self.uniform_weight,
            ensemble_probability=self.ensemble_probability,
            batch_size=self.batch_size,
            max_size=self.max_size,
            seed=self.seed,
            n_most_recent=self.n_most_recent,
            id=self.id,
        )


@config()
class MixedHistoryBufferConfig:
    name: Literal["mixed_history_buffer"] = "mixed_history_buffer"
    ensemble: int = MISSING
    max_size: int = 1_000_000
    ensemble_probability: float = 0.5
    batch_size: int = 256
    seed: int = 0
    n_most_recent: int = 1
    online_weight: float = 0.75
    id: str = ""

    @computed('ensemble')
    @classmethod
    def _ensemble(cls, cfg: "MainConfig"):
        return cfg.feature_flags.ensemble + cfg.agent.critic.rolling_reset_config.num_background_critics

    def to_lib_config(self) -> LibMixedHistoryBufferConfig:
        return LibMixedHistoryBufferConfig(
            name=self.name,
            ensemble=self.ensemble,
            max_size=self.max_size,
            ensemble_probability=self.ensemble_probability,
            batch_size=self.batch_size,
            seed=self.seed,
            n_most_recent=self.n_most_recent,
            online_weight=self.online_weight,
            id=self.id,
        )
