"""Shared fixtures for trajectory data used across multiple test modules."""

import datetime as dt

import jax.numpy as jnp
import pytest
from lib_agent.buffer.datatypes import Step, Trajectory
from lib_utils.named_array import NamedArray


@pytest.fixture
def trajectories_with_timestamps() -> list[Trajectory]:
    """Trajectories with timestamps. Values do not matter"""
    # Create timestamps
    base_time = dt.datetime(2023, 7, 13, 10, 0, tzinfo=dt.UTC)

    # Trajectory 1: Two-step trajectory with timestamps
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

    trajectory_1 = Trajectory(
        steps=[step_1_1, step_1_2],
        n_step_gamma=0.9,
        n_step_reward=1.5,
    )

    # Trajectory 2: Another two-step trajectory with different timestamps
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

    trajectory_2 = Trajectory(
        steps=[step_2_1, step_2_2],
        n_step_gamma=0.85,
        n_step_reward=1.2,
    )

    return [trajectory_1, trajectory_2]
