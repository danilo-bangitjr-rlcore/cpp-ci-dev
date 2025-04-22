from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Step:
    state: np.ndarray
    action: np.ndarray
    reward: float | None
    gamma: float


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    gamma: float
    next_state: np.ndarray

    @property
    def state_dim(self):
        return self.state.shape[-1]

    @property
    def action_dim(self):
        return self.action.shape[-1]


class TransitionCreator:
    def __init__(self, n_step: int, gamma: float = 1.0):
        self._gamma = gamma
        self._n_step = n_step

        self._buffer = deque[Step](maxlen=n_step + 1)

    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float | None,
        done: bool,
    ) -> list[Transition]:
        gamma = float(done) * self._gamma
        self._buffer.append(Step(
            state,
            action,
            reward,
            gamma,
        ))

        # if the reward is None, we know this is the first
        # step of the episode
        if reward is None:
            return []

        if not self._buffer_full():
            return []

        trans = _build_n_step_transitions(self._buffer, self._n_step)
        return [trans]

    def flush(self):
        self._buffer.clear()

    def _buffer_full(self):
        return len(self._buffer) == (self._n_step + 1)



def _build_n_step_transitions(buffer: deque[Step], n: int):
    ret = 0
    gamma = 1.0

    first = buffer[0]
    for i, step in enumerate(buffer):
        # deque doesn't support slicing like buffer[1:]
        if i == 0: continue
        if i == n+1: break

        r = step.reward
        assert r is not None
        ret += gamma * r
        gamma *= step.gamma

    last = buffer[n]

    return Transition(
        state=first.state,
        action=first.action,
        reward=ret,
        gamma=gamma,
        next_state=last.state,
    )
