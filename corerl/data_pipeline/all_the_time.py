import torch
import pandas as pd
import numpy as np
import math

from collections.abc import Iterable
from collections import deque
from dataclasses import dataclass

from corerl.data_pipeline.tag_config import TagConfig
from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, Step, Transition, StageCode
from corerl.configs.config import interpolate, config, MISSING


@config()
class AllTheTimeTCConfig:
    name: str = "all-the-time"
    gamma: float = interpolate('${experiment.gamma}')
    min_n_step: int = 1
    max_n_step: int = MISSING


@dataclass(init=False)
class NStepInfo:
    """
    Dataclass for holding on to information for producing transitions for each bootstrap length (n)
    Holds:
     * a queue of steps
     * the discounted sum of rewards for steps in step_q. The first step's reward is ignored since it occurred
        prior to the first step's state being created
     * the discount factor for bootstrapping off step_q[-1].state
    """

    def __init__(self, n: int):
        self.step_q = deque[Step](maxlen=n + 1)
        self.n_step_reward: float = 0
        self.n_step_gamma: float = 1


type StepInfo = dict[int, NStepInfo]


def has_nan(obj: object) -> bool:
    for _, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                return True
        elif isinstance(value, float) and math.isnan(value):
            return True
    return False


def get_tags(df: pd.DataFrame, tags: Iterable[str]) -> torch.Tensor:
    data_np = df[list(tags)].to_numpy().astype(np.float32)
    return tensor(data_np)


def get_n_step_reward(step_q: deque[Step]) -> tuple[float, float]:
    last_step = step_q[-1]
    n_step_reward = last_step.reward
    n_step_gamma = last_step.gamma

    for step_backwards in range(len(step_q) - 2, 0, -1):
        step = step_q[step_backwards]
        n_step_reward = step.reward + step.gamma * n_step_reward
        n_step_gamma *= step.gamma

    return n_step_reward, n_step_gamma


def _reset_step_info(min_n_step: int, max_n_step: int) -> StepInfo:
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
        self.meta_tags = sorted(tag.name for tag in tag_configs if tag.is_meta)

        self.gamma = cfg.gamma
        self.min_n_step = cfg.min_n_step
        self.max_n_step = cfg.max_n_step
        assert self.min_n_step > 0
        assert self.max_n_step >= self.min_n_step

    def _make_steps(self, pf: PipelineFrame) -> tuple[PipelineFrame, list[Step]]:
        """
        Constructs steps from pipeframe elements
        """
        pf, actions, states = self._extract_columns(pf)

        df = pf.data
        rewards = df['reward'].to_numpy()
        gammas = np.ones(len(rewards))
        if 'terminated' in df.columns:
            gammas = 1 - df['terminated'].to_numpy()

        dps = pf.decision_points
        assert len(actions) == len(states) and len(states) == len(rewards) == len(gammas) == len(dps)

        steps: list[Step] = []
        for i in range(len(actions)):
            step = Step(
                reward=rewards[i],
                action=actions[i],
                gamma=float(self.gamma * gammas[i]),
                state=states[i],
                dp=bool(dps[i]),
            )
            steps.append(step)
        return pf, steps

    def _extract_columns(self, pf: PipelineFrame) -> tuple[PipelineFrame, torch.Tensor, torch.Tensor]:
        state_cols = [
            col for col in pf.data.columns
            if col not in self.meta_tags and col != "reward"
        ]
        states = get_tags(pf.data, state_cols)
        actions = tensor(pf.actions.to_numpy(np.float32))
        return pf, actions, states

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        step_info = pf.temporal_state.get(
            StageCode.TC,
            _reset_step_info(self.min_n_step, self.max_n_step)
        )

        assert isinstance(step_info, dict)

        pf, steps = self._make_steps(pf)

        transitions = []
        for step in steps:
            if has_nan(step):
                step_info = _reset_step_info(self.min_n_step, self.max_n_step)  # nuke the step info
            else:
                new_transitions, step_info = self._update(step, step_info)
                transitions += new_transitions

        pf.temporal_state[StageCode.TC] = step_info
        pf.transitions = transitions

        return pf

    def _update(self, step: Step, step_info: StepInfo) -> tuple[list[Transition], StepInfo]:
        """
        Updates all the step queues, n_step_rewards, and n_step_gammas stored in self.step_info with the new step,
        then returns any produced transitions.
        """
        new_transitions = []
        for n in range(self.min_n_step, self.max_n_step + 1):
            n_step_info = step_info[n]
            step_q = n_step_info.step_q
            step_q.append(step)

            is_full = len(step_q) == n + 1
            if is_full:
                n_step_info.n_step_reward, n_step_info.n_step_gamma = get_n_step_reward(step_q)

                new_transition = Transition(
                    list(step_q),
                    n_step_reward=n_step_info.n_step_reward,
                    n_step_gamma=n_step_info.n_step_gamma,
                )

                new_transitions.append(new_transition)

        return new_transitions, step_info
