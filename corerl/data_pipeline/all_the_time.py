import torch
import pandas as pd
import numpy as np
import math

from collections import deque
from dataclasses import dataclass

from corerl.data_pipeline.tag_config import TagConfig
from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, Step, NewTransition, StageCode
from corerl.configs.config import interpolate


@dataclass
class AllTheTimeTCConfig:
    name: str = "all-the-time"
    gamma: float = interpolate('${experiment.gamma}')
    min_n_step: int = 1
    max_n_step: int = interpolate('${action_period}')


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


def get_tags(df: pd.DataFrame, tags: list[str] | str) -> torch.Tensor:
    data_np = df[tags].to_numpy().astype(np.float32)
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
        self._init_action_tags()

        self.gamma = cfg.gamma
        self.min_n_step = cfg.min_n_step
        self.max_n_step = cfg.max_n_step
        assert self.min_n_step > 0
        assert self.max_n_step >= self.min_n_step

    def _init_action_tags(self):
        self.action_tags = []
        for tag_config in self.tag_configs:
            name = tag_config.name
            if tag_config.is_action:
                self.action_tags.append(name)

    def _make_steps(self, pf: PipelineFrame) -> list[Step]:
        """
        Makes the steps for the pf
        """
        df = pf.data
        actions = get_tags(df, self.action_tags)
        state_tags = sorted(
            set(df.columns) - set(self.action_tags) - {'reward', 'trunc', 'term'}
        )
        states = get_tags(df, state_tags)
        rewards = df['reward'].to_numpy()
        gammas = np.ones(len(rewards))
        if 'term' in df.columns:
            gammas = 1 - df['term'].to_numpy()
        dps = pf.decision_points
        assert len(actions) == len(states) and len(states) == len(rewards) == len(gammas) == len(dps)

        steps = []
        for i in range(len(actions)):
            step = Step(
                reward=rewards[i],
                action=actions[i],
                gamma=float(self.gamma * gammas[i]),
                state=states[i],
                dp=bool(dps[i]),
            )
            steps.append(step)
        return steps

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        step_info = pf.temporal_state.get(
            StageCode.TC,
            _reset_step_info(self.min_n_step, self.max_n_step)
        )

        assert isinstance(step_info, dict)

        steps = self._make_steps(pf)
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

    def _update(self, step: Step, step_info: StepInfo) -> tuple[list[NewTransition], StepInfo]:
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

                assert n_step_info.n_step_reward is not None
                assert n_step_info.n_step_gamma is not None

                new_transition = NewTransition(
                    list(step_q),
                    n_step_reward=n_step_info.n_step_reward,
                    n_step_gamma=n_step_info.n_step_gamma,
                )

                new_transitions.append(new_transition)

        return new_transitions, step_info
