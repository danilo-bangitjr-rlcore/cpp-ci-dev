from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Step:
    state: np.ndarray
    a_lo: np.ndarray
    a_hi: np.ndarray
    action: np.ndarray
    reward: float | None
    gamma: float
    dp: bool


@dataclass
class Transition:
    state: np.ndarray
    a_lo: np.ndarray
    a_hi: np.ndarray
    dp: bool
    last_a: np.ndarray
    action: np.ndarray
    reward: float
    gamma: float
    next_state: np.ndarray
    next_a_lo: np.ndarray
    next_a_hi: np.ndarray
    next_dp: bool

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
        state=first.state,
        a_lo=first.a_lo,
        a_hi=first.a_hi,
        dp=first.dp,
        # use previous action if available, otherwise use default to action in the transition
        last_a=prev.action if prev is not None else first.action,
        action=first.action,
        reward=ret,
        gamma=gamma,
        next_state=last.state,
        next_a_lo=last.a_lo,
        next_a_hi=last.a_hi,
        next_dp=last.dp,
    )
