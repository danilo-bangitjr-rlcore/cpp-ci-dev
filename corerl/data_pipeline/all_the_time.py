from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from corerl.component.network.utils import tensor
from corerl.configs.config import MISSING, computed, config, interpolate
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, Step, Transition
from corerl.data_pipeline.tag_config import TagConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class AllTheTimeTCConfig:
    name: str = "all-the-time"
    gamma: float = interpolate('${experiment.gamma}')
    min_n_step: int = 1
    max_n_step: int = MISSING

    @computed('max_n_step')
    @classmethod
    def _max_n_step(cls, cfg: MainConfig):
        ap_sec = cfg.interaction.action_period.total_seconds()
        obs_sec = cfg.interaction.obs_period.total_seconds()

        steps_per_decision = int(ap_sec / obs_sec)
        assert np.isclose(steps_per_decision, ap_sec / obs_sec), \
            "Action period must be a multiple of obs period"

        return steps_per_decision



@dataclass(init=False)
class NStepInfo:
    """
    Dataclass for holding on to information for producing transitions for each bootstrap length (n)
    Holds:
     * a queue of steps
    """

    def __init__(self, n: int):
        self.step_q = deque[Step](maxlen=n + 1)

type StepInfo = dict[int, NStepInfo]


def get_tags(df: pd.DataFrame, tags: Iterable[str]):
    data_np = df[list(tags)].to_numpy().astype(np.float32)
    return tensor(data_np)


def get_n_step_reward(step_q: deque[Step]):
    steps = deepcopy(step_q) # deque is mutable
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
        n: NStepInfo(n) for n in range(min_n_step, max_n_step + 1)
    }
    return step_info

class AllTheTimeTC:
    def __init__(
            self,
            cfg: AllTheTimeTCConfig,
            tag_configs: list[TagConfig],
    ):
        self.cfg = cfg
        self.tag_configs = tag_configs

        self.gamma = cfg.gamma
        self.min_n_step = cfg.min_n_step
        self.max_n_step = cfg.max_n_step
        assert self.min_n_step > 0
        assert self.max_n_step >= self.min_n_step

    def _make_steps(self, pf: PipelineFrame) -> tuple[PipelineFrame, list[Step]]:
        """
        Constructs steps from pipeframe elements
        """

        states = tensor(pf.states.to_numpy())
        actions = tensor(pf.actions.to_numpy())
        rewards = pf.rewards['reward'].to_numpy()

        dps = pf.decision_points
        n = len(pf.data)

        assert n == len(actions) == len(states) and len(states) == len(rewards) == len(dps)

        steps: list[Step] = []
        for i in range(n):
            step = Step(
                reward=rewards[i],
                action=actions[i],
                gamma=self.gamma,
                state=states[i],
                dp=bool(dps[i]),
            )
            steps.append(step)
        return pf, steps

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        step_info = pf.temporal_state.get(
            StageCode.TC,
            _reset_step_info(self.min_n_step, self.max_n_step)
        )

        assert isinstance(step_info, dict)

        pf, steps = self._make_steps(pf)

        transitions = []
        for step in steps:
            new_transitions, step_info = self._update(step, step_info)
            transitions += new_transitions

        pf.temporal_state[StageCode.TC] = step_info
        pf.transitions = transitions

        return pf

    def _update(self, step: Step, step_info: StepInfo):
        """
        Updates all the step queues, n_step_rewards, and n_step_gammas stored in self.step_info with the new step,
        then returns any produced transitions.
        """
        new_transitions: list[Transition] = []
        for n in range(self.min_n_step, self.max_n_step + 1):
            n_step_info = step_info[n]
            step_q = n_step_info.step_q
            step_q.append(step)

            is_full = len(step_q) == step_q.maxlen
            if is_full:
                n_step_reward, n_step_gamma = get_n_step_reward(step_q)

                new_transition = Transition(
                    list(step_q),
                    n_step_reward=n_step_reward,
                    n_step_gamma=n_step_gamma,
                )

                new_transitions.append(new_transition)

        return new_transitions, step_info
