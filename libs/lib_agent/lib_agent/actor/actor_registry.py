from typing import Any

from lib_agent.actor.percentile_actor import PAConfig, PercentileActor


def get_actor(
    cfg: dict[str, Any],
    state_dim: int,
    action_dim: int,
):
    name = cfg['name']
    if name == 'Percentile':
        return PercentileActor(PAConfig(**cfg), state_dim, action_dim)

    raise NotImplementedError()
