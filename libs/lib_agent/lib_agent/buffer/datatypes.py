import datetime
from dataclasses import dataclass, fields
from enum import Enum, IntEnum, auto
from math import isclose
from typing import Any, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
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
    primitive_held: jax.Array
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
    primitive_held: jax.Array


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
            primitive_held=jnp.asarray(trajectory.prior.primitive_held),
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
            primitive_held=jnp.asarray(trajectory.post.primitive_held),
        ),
        timestamp=timestamp,
    )


##############################################
#                                            #
#                  Options                   #
#                                            #
##############################################

class SupportsJaxification(Protocol):
    @property
    def features(self) -> jax.Array: ...

class PrimitivePolicy(Protocol):
    def __call__(self, state: State) -> jax.Array: ...

class TerminationFunction(Protocol):
    def __call__(self, state: State) -> jax.Array: ...

class OptionType(IntEnum):
    hold_action = auto()
    ramp = auto()

class Option(Protocol):
    @property
    def opt_params(self) -> SupportsJaxification: ...
    @property
    def pi(self) -> PrimitivePolicy: ...
    @property
    def beta(self) -> TerminationFunction: ...

    @property
    def features(self) -> jax.Array: ...

class OptionParams_(SupportsJaxification, Protocol):
    def as_option(self) -> Option: ...


class HoldActionParams(NamedTuple):
    action: jax.Array
    duration: jax.Array

    @property
    def opt_type(self) -> OptionType:
        return OptionType.hold_action

    @property
    def features(self) -> jax.Array:
        return jnp.hstack([self.action, self.duration])


class RampParams(NamedTuple):
    action: jax.Array
    delta: jax.Array

    @property
    def opt_type(self) -> OptionType:
        return OptionType.ramp

    @property
    def features(self):
        return jnp.hstack([self.action, self.delta])

@register_pytree_node_class
class HoldActionOption:
    def __init__(self, opt_params: HoldActionParams):
        self.opt_params = opt_params
        self.duration = opt_params.duration

    @property
    def pi(self) -> PrimitivePolicy:
        return lambda state: self.opt_params.action

    @property
    def beta(self) -> TerminationFunction:
        """
        termination must only be evaluated when duration is a scalar Array:
        to check termination of multiple versions of this option in parallel,
        the caller must rely on vmap
        """
        chex.assert_shape(self.duration, ())
        return lambda state: state.primitive_held >= self.duration

    @property
    def features(self) -> jax.Array:
        return self.opt_params.features

    def tree_flatten(self):
        children = (self.opt_params,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[HoldActionParams,]): # TODO: annotations
        opt_params = children[0]
        return cls(opt_params)

@register_pytree_node_class
class RampOption:
    def __init__(self, opt_params: RampParams):
        self.opt_params = opt_params

    @property
    def pi(self) -> PrimitivePolicy:
        delta = self.opt_params.delta
        target_a = self.opt_params.action

        def ramp_to_target(state: State):
            prev_a = state.last_a
            return prev_a + jnp.clip(target_a - prev_a, -delta, delta)

        return ramp_to_target

    @property
    def beta(self) -> TerminationFunction:
        """
        termination must only be evaluated when action is a rank 1 Array:
        to check termination of multiple versions of this option in parallel,
        the caller must rely on vmap
        """
        chex.assert_rank(self.opt_params.action, 1)
        return lambda state: jnp.allclose(state.last_a, self.opt_params.action)

    @property
    def features(self) -> jax.Array:
        return self.opt_params.features

    def tree_flatten(self):
        children = (self.opt_params,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[RampParams,]): # TODO: annotations
        opt_params = children[0]
        return cls(opt_params)
