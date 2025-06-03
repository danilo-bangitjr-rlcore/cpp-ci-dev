from typing import Any

from ml_instrumentation.Collector import Collector

from lib_agent.actor.percentile_actor import PAConfig, PercentileActor


def get_actor(
    cfg: dict[str, Any],
    seed: int,
    state_dim: int,
    action_dim: int,
    collector: Collector,
):
    name = cfg['name']
    if name == 'Percentile':
        return PercentileActor(PAConfig(**cfg), seed, state_dim, action_dim, collector)

    raise NotImplementedError()
