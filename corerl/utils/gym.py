import gym
import gym.spaces

import gymnasium
import gymnasium.spaces
from typing import Any

def space_bounds(space: gym.Space[Any] | gymnasium.Space[Any]):
    assert isinstance(
        space,
        (gym.spaces.Box, gymnasium.spaces.Box),
    )
    return (
        space.low,
        space.high,
    )
