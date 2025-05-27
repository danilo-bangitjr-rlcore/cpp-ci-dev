from typing import Any

import gymnasium
import gymnasium.spaces


def space_bounds(space: gymnasium.Space[Any]):
    assert isinstance(space, gymnasium.spaces.Box)
    return (
        space.low,
        space.high,
    )

def space_shape(space: gymnasium.Space[Any]):
    assert isinstance(space, gymnasium.spaces.Box)
    return space.shape
