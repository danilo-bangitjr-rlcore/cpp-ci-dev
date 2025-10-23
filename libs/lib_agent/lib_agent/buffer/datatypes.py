import datetime
from dataclasses import dataclass, fields
from enum import Enum, auto
from math import isclose
from typing import NamedTuple

import jax
import jax.numpy as jnp
from lib_utils.named_array import NamedArray


class DataMode(Enum):
    OFFLINE = auto()
    ONLINE = auto()
    REFRESH = auto()


@dataclass
class Step:
    """
    Dataclass for storing the information of a single step.
    At least two of these make up a Trajectory.
    """
    reward: float
    action: jax.Array
    gamma: float
    state: NamedArray
    action_lo: jax.Array
    action_hi: jax.Array
    dp: bool # decision point
    ac: bool # action change
    timestamp: datetime.datetime | None = None

    def __eq__(self, other: object):
        if not isinstance(other, Step):
            return False

        return (
                isclose(self.gamma, other.gamma)
                and isclose(self.reward, other.reward)
                and jnp.allclose(self.action, other.action).item()
                and jnp.allclose(self.state.array, other.state.array).item()
                and self.dp == other.dp
        )

    def __str__(self):
        return '\n'.join(
            f'{f.name}: {getattr(self, f.name)}'
            for f in fields(self)
        )

    def __hash__(self):
        return hash((
            self.reward,
            tuple(self.action),
            self.gamma,
            tuple(self.state.array),
            self.action_lo,
            self.action_hi,
            self.dp,
            self.ac,
            self.timestamp,
        ))


@dataclass
class Trajectory:
    steps: list[Step]
    n_step_reward: float
    n_step_gamma: float

    @property
    def state(self):
        return self.prior.state

    @property
    def action(self):
        return self.post.action

    @property
    def reward(self):
        return self.n_step_reward

    @property
    def gamma(self):
        return self.n_step_gamma

    @property
    def next_state(self):
        return self.post.state

    @property
    def action_dim(self):
        return self.post.action.shape[-1]

    @property
    def state_dim(self):
        return self.prior.state.shape[-1]

    @property
    def prior(self):
        return self.steps[0]

    @property
    def post(self):
        return self.steps[-1]

    @property
    def n_steps(self):
        return len(self.steps) - 1

    def __eq__(self, other: object):
        if not isinstance(other, Trajectory):
            return False

        if len(self.steps) != len(other.steps):
            return False

        if self.n_steps != other.n_steps:
            return False

        if self.n_step_gamma != other.n_step_gamma:
            return False

        for i, step in enumerate(self.steps):
            if step != other.steps[i]:
                return False

        return True

    def __len__(self) -> int:
        return len(self.steps)-1

    def __hash__(self):
        return hash((
            tuple(self.steps),
            self.n_step_reward,
            self.n_step_gamma,
        ))

    @property
    def start_time(self):
        return self.prior.timestamp

    @property
    def end_time(self):
        return self.post.timestamp


class State(NamedTuple):
    features: NamedArray
    a_lo : jax.Array
    a_hi : jax.Array
    dp: jax.Array
    last_a: jax.Array


class Transition(NamedTuple):
    state: State
    action: jax.Array
    n_step_reward: jax.Array
    n_step_gamma: jax.Array
    next_state: State
    timestamp: int | None = None


def convert_trajectory_to_transition(trajectory: Trajectory) -> Transition:
    timestamp = None
    if trajectory.start_time is not None:
        timestamp = int(trajectory.start_time.timestamp())

    return Transition(
        state=State(
            features=trajectory.state,
            a_lo=trajectory.prior.action_lo,
            a_hi=trajectory.prior.action_hi,
            dp=jnp.expand_dims(trajectory.prior.dp, -1),
            last_a=trajectory.prior.action,
        ),
        action=trajectory.action,
        n_step_reward=jnp.asarray(trajectory.n_step_reward),
        n_step_gamma=jnp.asarray(trajectory.n_step_gamma),
        next_state=State(
            features=trajectory.next_state,
            a_lo=trajectory.post.action_lo,
            a_hi=trajectory.post.action_hi,
            dp=jnp.expand_dims(trajectory.post.dp, -1),
            last_a=trajectory.action,
        ),
        timestamp=timestamp,
    )
