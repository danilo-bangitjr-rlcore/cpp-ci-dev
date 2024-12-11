import warnings
import torch
import pandas as pd
import math

from collections import deque
from dataclasses import dataclass

from corerl.data_pipeline.tag_config import TagConfig
from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, Step, NewTransition
from corerl.utils.hydra import interpolate
from corerl.data_pipeline.transition_creators.base import (
    BaseTransitionCreator,
    BaseTransitionCreatorConfig,
    transition_creator_group,
    TransitionCreatorTemporalState,
)


@dataclass
class AllTheTimeTCConfig(BaseTransitionCreatorConfig):
    name: str = "all-the-time"
    gamma: float = interpolate('${experiment.gamma}')
    min_n_step: int = 1
    max_n_step: int = interpolate('${interaction.steps_per_decision}')

    # FILTERS
    only_dp_transitions: bool = False  # whether we only want to return transitions between decision points


class NStepInfo:
    def __init__(self, n: int):
        self.step_q = deque(maxlen=n + 1)
        self.n_step_reward: int | None = None
        self.n_step_gamma: int | None = None


@dataclass
class AllTheTimeTS(TransitionCreatorTemporalState):
    step_info: dict[int, NStepInfo]


def has_nan(obj):
    for attr_name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                return True
        elif isinstance(value, float) and math.isnan(value):
            return True
    return False


def get_tags(df: pd.DataFrame, tags: list[str] | str) -> torch.Tensor:
    return tensor(df[tags].to_numpy())


def get_n_step_reward(step_q):
    last_step = step_q[-1]
    n_step_reward = last_step.reward
    n_step_gamma = last_step.gamma

    for step_backwards in range(len(step_q) - 2, 0, -1):
        step = step_q[step_backwards]
        n_step_reward = step.reward + step.gamma * n_step_reward
        n_step_gamma *= step.gamma

    return n_step_reward, n_step_gamma


def update_n_step_reward_gamma(n_step_reward, n_step_gamma, step_q):
    assert (n_step_reward is None) == (n_step_gamma is None)

    if n_step_gamma is None:
        n_step_reward, n_step_gamma = get_n_step_reward(step_q)

    else:  # update n_step_reward online with a recurrence relation
        assert n_step_reward is not None
        discount = n_step_gamma / step_q[-1].gamma
        n_step_gamma = discount * step_q[0].gamma
        n_step_reward = (n_step_reward - step_q[-1].reward) / step_q[-1].gamma
        n_step_reward = n_step_reward + discount * step_q[0].reward

    return n_step_reward, n_step_gamma


def make_transition(
        step_q: deque[Step],
        n_step_reward: float | None = None,
        n_step_gamma: float | None = None) -> tuple[NewTransition, float, float]:
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, step_q)

    pre_step = step_q[0]
    post_step = Step(
        reward=n_step_reward,
        action=step_q[-1].action,
        gamma=n_step_gamma,
        state=step_q[-1].state,
        dp=False,  #TODO: set this
    )

    transition = NewTransition(
        pre_step,
        post_step,
        n_steps=len(step_q) - 1
    )

    return transition, n_step_reward, n_step_gamma


class AllTheTimeTC(BaseTransitionCreator):
    def __init__(
            self,
            cfg: AllTheTimeTCConfig,
            tag_configs: list[TagConfig],
    ):
        super().__init__(cfg, tag_configs)

        self.gamma = cfg.gamma
        self.min_n_step = cfg.min_n_step
        self.max_n_step = cfg.max_n_step
        self.step_info = None
        self.init_step_info()

        self.only_dp_transitions = cfg.only_dp_transitions

    def init_step_info(self):
        self.step_info: dict[int, NStepInfo] = {
            n: NStepInfo(n) for n in range(self.min_n_step, self.max_n_step + 1)
        }

    def _inner_call(self,
                    pf: PipelineFrame,
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[NewTransition], TransitionCreatorTemporalState | None]:

        assert isinstance(tc_ts, AllTheTimeTS | None)

        transitions = []

        if pf.data.empty:  # pretend like nothing happened but raise a warning...
            warnings.warn("Empty dataframe passed to transition creator", stacklevel=2)
            return transitions, tc_ts

        df = pf.data
        actions = get_tags(df, self.action_tags)
        state_tags = sorted(
            set(df.columns) - set(self.action_tags + ['reward'])
        )
        states = get_tags(df, state_tags)
        rewards = df['reward'].to_numpy()
        assert len(actions) == len(states) and len(states) == len(rewards)

        for i in range(len(actions)):
            step = Step(
                reward=rewards[i],
                action=actions[i],
                gamma=self.gamma,
                state=states[i],
                dp=False,
            )

            if has_nan(step):
                self.init_step_info()  # nuke the step info
            else:
                transitions += self._update(step)

        if tc_ts is None:
            tc_ts = AllTheTimeTS(self.step_info)
        else:
            tc_ts.step_info = self.step_info

        return transitions, tc_ts

    def _update(self, step: Step) -> list[NewTransition]:
        new_transitions = []
        for n in range(self.min_n_step, self.max_n_step + 1):
            n_step_info = self.step_info[n]
            step_q = n_step_info.step_q
            step_q.append(step)

            is_full = len(step_q) == n + 1
            if is_full:
                n_step_reward = n_step_info.n_step_reward
                n_step_gamma = n_step_info.n_step_gamma
                new_transition, n_step_reward, n_step_gamma = make_transition(step_q, n_step_reward, n_step_gamma)
                new_transitions.append(new_transition)
                n_step_info.n_step_reward = n_step_reward
                n_step_info.n_step_gamma = n_step_gamma

        return new_transitions


transition_creator_group.dispatcher(AllTheTimeTC)
