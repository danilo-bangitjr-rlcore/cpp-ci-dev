from typing import Annotated

from lib_config.group import Group
from pydantic import Field

from lib_agent.buffer.mixed_history_buffer import MixedHistoryBuffer, MixedHistoryBufferConfig
from lib_agent.buffer.recency_bias_buffer import RecencyBiasBuffer, RecencyBiasBufferConfig

BufferConfig = Annotated[(
    MixedHistoryBufferConfig
    | RecencyBiasBufferConfig
), Field(discriminator='name')]

buffer_group = Group[
    [],
    MixedHistoryBuffer | RecencyBiasBuffer,
]()


def create_mixed_history_buffer(cfg: MixedHistoryBufferConfig) -> MixedHistoryBuffer:
    return MixedHistoryBuffer(
        n_ensemble=cfg.n_ensemble,
        max_size=cfg.max_size,
        ensemble_prob=cfg.ensemble_prob,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        n_most_recent=cfg.n_most_recent,
        online_weight=cfg.online_weight,
        id=cfg.id,
    )


def create_recency_bias_buffer(cfg: RecencyBiasBufferConfig) -> RecencyBiasBuffer:
    return RecencyBiasBuffer(cfg)


buffer_group.dispatcher(create_mixed_history_buffer)
buffer_group.dispatcher(create_recency_bias_buffer)
