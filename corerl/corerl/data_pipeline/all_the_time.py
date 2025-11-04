from __future__ import annotations

from collections import deque

import jax.numpy as jnp
import numpy as np
import pandas as pd
from lib_agent.buffer.datatypes import Step, Trajectory
from lib_utils.maybe import Maybe
from lib_utils.named_array import NamedArray

from corerl.configs.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode

type StepInfo = dict[int, deque[Step]]


def get_n_step_reward(step_q: deque[Step]):
    steps = step_q.copy() # deque is mutable
    steps.popleft() # drop the first step, it does not contribute to return

    partial_return = 0
    discount = 1

    while len(steps) > 0:
        step = steps.popleft()
        partial_return += discount * step.reward
        discount *= step.gamma

    return partial_return, discount


def _reset_step_info(min_n_step: int, max_n_step: int):
    step_info: StepInfo = {
        n: deque[Step](maxlen=n+1) for n in range(min_n_step, max_n_step + 1)
    }
    return step_info

class AllTheTimeTC:
    def __init__(
            self,
            cfg: AllTheTimeTCConfig,
    ):
        self.cfg = cfg

        self.gamma = cfg.gamma
        self.return_scale = cfg.return_scale
        self.min_n_step = cfg.min_n_step
        self.max_n_step = cfg.max_n_step

    def _make_steps(self, pf: PipelineFrame) -> tuple[PipelineFrame, list[Step]]:
        """
        Constructs steps from pipeframe elements
        """

        states = NamedArray.from_pandas(pf.states)
        actions = jnp.asarray(pf.actions.to_numpy(dtype=np.float32))
        rewards = pf.rewards['reward'].to_numpy(dtype=np.float32).copy()
        if self.cfg.normalize_return:
            rewards *= self.return_scale*(1 - self.gamma)

        # dynamic action bounds
        action_lo = jnp.asarray(pf.action_lo.to_numpy(dtype=np.float32))
        action_hi = jnp.asarray(pf.action_hi.to_numpy(dtype=np.float32))

        dps = pf.decision_points
        acs = pf.action_change

        n = len(pf.data)
        assert n == len(actions) == len(states) and len(states) == len(rewards) == len(dps)

        steps: list[Step] = []
        for i in range(n):
            step = Step(
                reward=rewards[i],
                action=actions[i],
                gamma=self.gamma,
                state=states[i],
                action_lo=action_lo[i],
                action_hi=action_hi[i],
                dp=bool(dps[i]),
                ac=bool(acs[i]),
                primitive_held=pf.primitive_held[i],
                timestamp=( # curse you pandas
                    Maybe(pf.data.index[i])
                    .is_instance(pd.Timestamp)
                    .map(lambda x: x.to_pydatetime())
                    .unwrap()
                ),
            )
            steps.append(step)
        return pf, steps

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        step_info = pf.temporal_state.get(
            StageCode.TC,
            _reset_step_info(self.min_n_step, self.max_n_step),
        )

        assert isinstance(step_info, dict)

        pf, steps = self._make_steps(pf)

        trajectories = []
        for step in steps:
            new_trajectories, step_info = self._update(step, step_info)
            trajectories += new_trajectories

        pf.temporal_state[StageCode.TC] = step_info
        pf.trajectories = trajectories

        return pf

    def _update(self, step: Step, step_info: StepInfo):
        """
        Updates all the step queues, n_step_rewards, and n_step_gammas stored in self.step_info with the new step,
        then returns any produced trajectories.
        """
        new_trajectories: list[Trajectory] = []
        for n in range(self.min_n_step, self.max_n_step + 1):
            step_q = step_info[n]
            step_q.append(step)

            is_full = len(step_q) == step_q.maxlen
            if is_full:
                n_step_reward, n_step_gamma = get_n_step_reward(step_q)

                new_trajectory = Trajectory(
                    list(step_q),
                    n_step_reward=n_step_reward,
                    n_step_gamma=n_step_gamma,
                )

                new_trajectories.append(new_trajectory)

        return new_trajectories, step_info
