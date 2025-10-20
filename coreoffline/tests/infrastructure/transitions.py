"""Shared fixtures for transition data used across multiple test modules."""

import datetime as dt

import jax.numpy as jnp
import pytest
from corerl.data_pipeline.datatypes import Step, Transition
from lib_utils.named_array import NamedArray


@pytest.fixture
def transitions_with_timestamps() -> list[Transition]:
    """Transitions with timestamps. Values do not matter"""
    # Create timestamps
    base_time = dt.datetime(2023, 7, 13, 10, 0, tzinfo=dt.UTC)

    # Transition 1: Two-step transition with timestamps
    step_1_1 = Step(
        state=NamedArray.unnamed(jnp.array([1.0])),
        action=jnp.array([0.5]),
        reward=1.0,
        gamma=0.9,
        ac=False,
        dp=True,
        action_lo=jnp.array([0.0]),
        action_hi=jnp.array([1.0]),
        timestamp=base_time,
    )
    step_1_2 = Step(
        state=NamedArray.unnamed(jnp.array([2.0])),
        action=jnp.array([1.0]),
        reward=0.5,
        gamma=0.9,
        ac=True,
        dp=False,
        action_lo=jnp.array([0.0]),
        action_hi=jnp.array([1.0]),
        timestamp=base_time + dt.timedelta(minutes=1),
    )

    transition_1 = Transition(
        steps=[step_1_1, step_1_2],
        n_step_gamma=0.9,
        n_step_reward=1.5,
    )

    # Transition 2: Another two-step transition with different timestamps
    step_2_1 = Step(
        state=NamedArray.unnamed(jnp.array([3.0])),
        action=jnp.array([0.0]),
        reward=0.0,
        gamma=0.85,
        ac=False,
        dp=True,
        action_lo=jnp.array([0.0]),
        action_hi=jnp.array([1.0]),
        timestamp=base_time + dt.timedelta(minutes=2),
    )
    step_2_2 = Step(
        state=NamedArray.unnamed(jnp.array([4.0])),
        action=jnp.array([0.8]),
        reward=1.2,
        gamma=0.85,
        ac=True,
        dp=True,
        action_lo=jnp.array([0.0]),
        action_hi=jnp.array([1.0]),
        timestamp=base_time + dt.timedelta(minutes=3),
    )

    transition_2 = Transition(
        steps=[step_2_1, step_2_2],
        n_step_gamma=0.85,
        n_step_reward=1.2,
    )

    return [transition_1, transition_2]
