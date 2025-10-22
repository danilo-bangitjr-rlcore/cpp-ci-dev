from collections import deque
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from lib_agent.buffer.datatypes import State, Transition
from lib_utils.named_array import NamedArray


@dataclass
class Step:
    state: np.ndarray
    a_lo: np.ndarray
    a_hi: np.ndarray
    action: np.ndarray
    reward: float | None
    gamma: float
    dp: bool


class TransitionCreator:
    def __init__(self, n_step: int, gamma: float = 1.0):
        self._gamma = gamma
        self._n_step = n_step

        self._buffer = deque[Step | None](maxlen=n_step + 2)

    def __call__(
        self,
        state: np.ndarray,
        a_lo: np.ndarray,
        a_hi: np.ndarray,
        action: np.ndarray,
        reward: float | None,
        done: bool,
        dp: bool,
    ) -> list[Transition]:
        gamma = (1 - float(done)) * self._gamma

        self._buffer.append(Step(
            state,
            a_lo,
            a_hi,
            action,
            reward,
            gamma,
            dp,
        ))

        # if the reward is None, we know this is the first
        # step of the episode
        if reward is None:
            self._buffer.appendleft(None)
            return []

        if not self._buffer_full():
            return []

        trans = _build_n_step_transitions(self._buffer, self._n_step)
        return [trans]

    def flush(self):
        self._buffer.clear()

    def _buffer_full(self):
        return len(self._buffer) == (self._n_step + 2)

def _build_n_step_transitions(buffer: deque[Step | None], n: int):
    ret = 0
    gamma = 1.0

    prev = buffer[0] # step that happened previous to the first step in the transition
    first = buffer[1] # first step in the transition

    assert isinstance(first, Step), "First step in the transition must not be None"
    for i, step in enumerate(buffer):
        # deque doesn't support slicing like buffer[1:]
        if i < 2: continue
        if i == n+2: break

        assert step is not None
        r = step.reward
        assert r is not None
        ret += gamma * r
        gamma *= step.gamma

    last = buffer[-1]
    assert isinstance(last, Step), "Last step in the transition must not be None"

    return Transition(
        state=State(
            features=NamedArray.unnamed(jnp.array(first.state)),
            a_lo=jnp.array(first.a_lo),
            a_hi=jnp.array(first.a_hi),
            dp=jnp.array([first.dp]),
            last_a=jnp.array(prev.action) if prev is not None else jnp.array(first.action),
        ),
        action=jnp.array(first.action),
        n_step_reward=jnp.array(ret),
        n_step_gamma=jnp.array(gamma),
        next_state=State(
            features=NamedArray.unnamed(jnp.array(last.state)),
            a_lo=jnp.array(last.a_lo),
            a_hi=jnp.array(last.a_hi),
            dp=jnp.array([last.dp]),
            last_a=jnp.array(last.action),
        ),
    )
