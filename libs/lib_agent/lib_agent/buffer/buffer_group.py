from typing import Annotated

from pydantic import Field

from lib_agent.buffer.mixed_history_buffer import MixedHistoryBuffer, MixedHistoryBufferConfig
from lib_agent.buffer.recency_bias_buffer import RecencyBiasBuffer, RecencyBiasBufferConfig

BufferConfig = Annotated[(
    MixedHistoryBufferConfig
    | RecencyBiasBufferConfig
), Field(discriminator='name')]


def buffer_group_dispatch(cfg: BufferConfig):
    if cfg.name == "mixed_history_buffer":
        return MixedHistoryBuffer(
            n_ensemble=cfg.n_ensemble,
            max_size=cfg.max_size,
            ensemble_probability=cfg.ensemble_probability,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            n_most_recent=cfg.n_most_recent,
            online_weight=cfg.online_weight,
            id=cfg.id,
        )
    if cfg.name == "recency_bias_buffer":
        return RecencyBiasBuffer(
            obs_period=cfg.obs_period,
            gamma=cfg.gamma,
            effective_episodes=cfg.effective_episodes,
            n_ensemble=cfg.n_ensemble,
            uniform_weight=cfg.uniform_weight,
            ensemble_probability=cfg.ensemble_probability,
            batch_size=cfg.batch_size,
            max_size=cfg.max_size,
            seed=cfg.seed,
            n_most_recent=cfg.n_most_recent,
            id=cfg.id,
        )
    raise ValueError(f"Unknown buffer type: {cfg.name}")


buffer_group = type('BufferGroup', (), {'dispatch': buffer_group_dispatch})()
