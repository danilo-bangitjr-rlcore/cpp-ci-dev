import math
from collections.abc import Iterable
from typing import Literal, assert_never

import jax
import jax.numpy as jnp
from lib_agent.buffer.datatypes import Trajectory
from lib_utils.named_array import NamedArray

from corerl.configs.data_pipeline.trajectory_filter import TrajectoryFilterConfig
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.state import AppState

type TrajectoryFilterType = (
    Literal['only_dp', 'only_no_action_change', 'only_post_dp', 'no_nan', 'only_pre_dp_or_ac']
)


class TrajectoryFilter:
    def __init__(self, app_state: AppState, cfg: TrajectoryFilterConfig):
        self.filter_names = cfg.filters
        self._app_state = app_state

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if pf.trajectories is None:
            return pf

        trajectories_before = len(pf.trajectories)
        for filter_name in self.filter_names:
            pf.trajectories = call_filter(pf.trajectories, filter_name)

            trajectories_after = len(pf.trajectories)
            self._app_state.metrics.write(
                self._app_state.agent_step,
                f'trajectories_filtered_by_{filter_name}',
                trajectories_before - trajectories_after,
            )
            trajectories_before = trajectories_after

        return pf


def call_filter(trajectories: Iterable[Trajectory], filter_name: TrajectoryFilterType):
    if filter_name == 'only_dp':
        trajectory_filter = only_dp
    elif filter_name == 'only_no_action_change':
        trajectory_filter = only_no_action_change
    elif filter_name == 'only_post_dp':
        trajectory_filter = only_post_dp
    elif filter_name == 'no_nan':
        trajectory_filter = no_nan
    elif filter_name == 'only_pre_dp_or_ac':
        trajectory_filter = only_pre_dp_or_ac
    else:
        assert_never(filter_name)

    return list(filter(trajectory_filter, trajectories))


def only_dp(trajectory: Trajectory):
    return trajectory.prior.dp and trajectory.post.dp


def only_pre_dp_or_ac(trajectory: Trajectory):
    return trajectory.prior.dp or trajectory.steps[1].ac


def only_post_dp(trajectory: Trajectory):
    return trajectory.post.dp


def only_no_action_change(trajectory: Trajectory):
    """
    The initial action change typically occurs on index 1:
        The agent takes action trajectory.steps[1].action in response
        to the state from trajectory.steps[0].state.
    This function checks for action changes after index 1.
    """
    i = 2 # check for action changes after initial action
    while i < len(trajectory.steps):
        # trust the countdown creator
        # NOTE: otherwise, we need to deal with delta actions correctly
        if trajectory.steps[i].ac:
            return False
        i += 1

    return True


def has_nan(obj: object):
    for value in vars(obj).values():
        if isinstance(value, jax.Array):
            if jnp.isnan(value).any():
                return True
        elif isinstance(value, NamedArray):
            if jnp.isnan(value.array).any():
                return True
        elif isinstance(value, float) and math.isnan(value):
            return True
    return False


def no_nan(trajectory: Trajectory):
    return not any(has_nan(step) for step in trajectory.steps)
