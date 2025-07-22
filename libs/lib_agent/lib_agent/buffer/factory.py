from typing import NamedTuple, TypeVar

from lib_agent.buffer.mixed_history_buffer import MixedHistoryBuffer, MixedHistoryBufferConfig
from lib_agent.buffer.recency_bias_buffer import RecencyBiasBuffer, RecencyBiasBufferConfig

BufferConfig = MixedHistoryBufferConfig | RecencyBiasBufferConfig

# NOTE: the python 3.12+ syntax for generic types is not compatible with pickle
T = TypeVar('T', bound=NamedTuple)


def build_buffer( # noqa: UP047
    cfg: BufferConfig,
    transition_type: type[T],
) -> MixedHistoryBuffer[T] | RecencyBiasBuffer[T]:

    if cfg.name == "mixed_history_buffer":
        return MixedHistoryBuffer[T](
            ensemble=cfg.ensemble,
            max_size=cfg.max_size,
            ensemble_probability=cfg.ensemble_probability,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            n_most_recent=cfg.n_most_recent,
            online_weight=cfg.online_weight,
            id=cfg.id,
        )
    if cfg.name == "recency_bias_buffer":
        return RecencyBiasBuffer[T](
            obs_period=cfg.obs_period,
            gamma=cfg.gamma,
            effective_episodes=cfg.effective_episodes,
            ensemble=cfg.ensemble,
            uniform_weight=cfg.uniform_weight,
            ensemble_probability=cfg.ensemble_probability,
            batch_size=cfg.batch_size,
            max_size=cfg.max_size,
            seed=cfg.seed,
            n_most_recent=cfg.n_most_recent,
            id=cfg.id,
        )
    raise ValueError(f"Unknown buffer type: {cfg.name}")
