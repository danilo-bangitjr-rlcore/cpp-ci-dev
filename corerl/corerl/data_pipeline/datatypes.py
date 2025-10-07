import datetime
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from math import isclose
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from lib_agent.buffer.buffer import State
from lib_agent.buffer.datatypes import DataMode, JaxTransition

type TagName = str  # alias to clarify semantics of PipelineStage and stage dict
type PipelineStage[T] = Callable[[T, TagName], T]

@dataclass
class Step:
    """
    Dataclass for storing the information of a single step.
    Two of these make up a transition.
    """
    reward: float
    action: jax.Array
    gamma: float
    state: jax.Array
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
                and jnp.allclose(self.state, other.state).item()
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
            tuple(self.state),
            self.action_lo,
            self.action_hi,
            self.dp,
            self.ac,
            self.timestamp,
        ))


@dataclass
class Transition:
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
        if not isinstance(other, Transition):
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


class AbsTransition(NamedTuple):
    state: State
    action: jax.Array
    reward: jax.Array
    next_state: State
    gamma: jax.Array


class StageCode(Enum):
    INIT = auto()
    VIRTUAL = auto()
    TRIGGER = auto()
    BOUNDS = auto()
    PREPROCESS = auto()
    IMPUTER = auto()
    ODDITY = auto()
    TC = auto()
    AC = auto()
    SC = auto()
    RC = auto()
    TF = auto()
    ZONES = auto()


type TemporalState = dict[StageCode, object | None]


@dataclass
class PipelineFrame:
    data: pd.DataFrame
    data_mode: DataMode
    last_stage : StageCode | None = None
    states: pd.DataFrame = field(init=False)
    actions: pd.DataFrame = field(init=False)
    action_lo: pd.DataFrame = field(init=False)
    action_hi: pd.DataFrame = field(init=False)
    rewards: pd.DataFrame = field(default_factory=pd.DataFrame)
    decision_points: np.ndarray = field(init=False)
    action_change: np.ndarray = field(init=False)
    temporal_state: TemporalState = field(default_factory=dict)
    transitions: list[Transition] | None = None

    def __post_init__(self):
        N = len(self.data)

        # initialize dp flags
        self.decision_points = np.zeros(N, dtype=np.bool_)
        # initialize action change flags
        self.action_change = np.zeros(N, dtype=np.bool_)

        # initialize rl containers
        self.states = self.data.copy(deep=False)
        self.actions = self.data.copy(deep=False)

    def get_last_timestamp(self) -> datetime.datetime:

        last_index = self.data.index[-1]
        assert isinstance(last_index, datetime.datetime)
        return last_index

    def get_first_timestamp(self) -> datetime.datetime:

        first_index = self.data.index[0]
        assert isinstance(first_index, datetime.datetime)
        return first_index

def convert_corerl_transition_to_jax_transition(corerl_transition: Transition) -> JaxTransition:
    return JaxTransition(
        last_action=corerl_transition.prior.action,
        state=corerl_transition.state,
        action=corerl_transition.action,
        reward=jnp.asarray(corerl_transition.reward),
        next_state=corerl_transition.next_state,
        gamma=jnp.asarray(corerl_transition.gamma),
        action_lo=corerl_transition.prior.action_lo,
        action_hi=corerl_transition.prior.action_hi,
        next_action_lo=corerl_transition.post.action_lo,
        next_action_hi=corerl_transition.post.action_hi,
        dp=jnp.asarray(corerl_transition.prior.dp),
        next_dp=jnp.asarray(corerl_transition.post.dp),
        n_step_reward=jnp.asarray(corerl_transition.n_step_reward),
        n_step_gamma=jnp.asarray(corerl_transition.n_step_gamma),
    )
