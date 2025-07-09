from typing import Annotated, Any, NamedTuple

from pydantic import Field

from lib_agent.buffer.mixed_history_buffer import MixedHistoryBuffer, MixedHistoryBufferConfig
from lib_agent.buffer.recency_bias_buffer import RecencyBiasBuffer, RecencyBiasBufferConfig

BufferConfig = Annotated[MixedHistoryBufferConfig | RecencyBiasBufferConfig | Any, Field(discriminator='name')]


def _to_lib_config(cfg: Any) -> MixedHistoryBufferConfig | RecencyBiasBufferConfig:
    if hasattr(cfg, 'to_lib_config'):
        return cfg.to_lib_config()
    return cfg


def build_buffer[T: NamedTuple](cfg: Any, transition_type: type[T]):
    lib_cfg = _to_lib_config(cfg)

    if lib_cfg.name == "mixed_history_buffer":
        return MixedHistoryBuffer[T](
            ensemble=lib_cfg.ensemble,
            max_size=lib_cfg.max_size,
            ensemble_probability=lib_cfg.ensemble_probability,
            batch_size=lib_cfg.batch_size,
            seed=lib_cfg.seed,
            n_most_recent=lib_cfg.n_most_recent,
            online_weight=lib_cfg.online_weight,
            id=lib_cfg.id,
        )
    if lib_cfg.name == "recency_bias_buffer":
        return RecencyBiasBuffer[T](
            obs_period=lib_cfg.obs_period,
            gamma=lib_cfg.gamma,
            effective_episodes=lib_cfg.effective_episodes,
            ensemble=lib_cfg.ensemble,
            uniform_weight=lib_cfg.uniform_weight,
            ensemble_probability=lib_cfg.ensemble_probability,
            batch_size=lib_cfg.batch_size,
            max_size=lib_cfg.max_size,
            seed=lib_cfg.seed,
            n_most_recent=lib_cfg.n_most_recent,
            id=lib_cfg.id,
        )
    raise ValueError(f"Unknown buffer type: {lib_cfg.name}")
