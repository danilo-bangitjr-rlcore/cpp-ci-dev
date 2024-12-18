import warnings
import torch
import pandas as pd
import numpy as np
import math

from collections import deque
from dataclasses import dataclass

from corerl.data_pipeline.tag_config import TagConfig
from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, Step, NewTransition, StageCode
from corerl.utils.hydra import interpolate


@dataclass
class AllTheTimeTCConfig:
    name: str = "all-the-time"
    gamma: float = interpolate('${experiment.gamma}')
    min_n_step: int = 1
    max_n_step: int = interpolate('${interaction.steps_per_decision}')


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
        self.n_step_reward: float | None = None
        self.n_step_gamma: float | None = None


@dataclass
class AllTheTimeTS:
    step_info: dict[int, NStepInfo]


def has_nan(obj: object):
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
    """
    Updates the n_step_reward and n_step_gamma as new steps are added to the queue.
    * step_q has the oldest transitions first (i.e. happened first).
    * If n_step_reward, n_step_gamma are None, initialize them by iterating backwards through the queue
        (get_n_step_reward). Otherwise, update n_step_reward and n_step_gamma recursively.
    """
    assert (n_step_reward is None) == (n_step_gamma is None)

    if n_step_gamma is None:
        n_step_reward, n_step_gamma = get_n_step_reward(step_q)

    else:  # update n_step_reward online with a recurrence relation
        assert n_step_reward is not None
        latest_reward_discount = n_step_gamma / step_q[0].gamma
        # subtract out the first (oldest) reward from the n_step_reward
        n_step_reward = n_step_reward - step_q[0].reward
        # Then un-discount all remaining rewards in the sum by the oldest discount factor
        n_step_reward = n_step_reward / step_q[0].gamma
        # discount and add the latest_reward to the sum
        n_step_reward = n_step_reward + latest_reward_discount * step_q[-1].reward
        # produce the gamma for discounting after the final step
        n_step_gamma = latest_reward_discount * step_q[-1].gamma

    return n_step_reward, n_step_gamma


class AllTheTimeTC:
    def __init__(
            self,
            cfg: AllTheTimeTCConfig,
            tag_configs: list[TagConfig],
    ):
        self.cfg = cfg
        self.stage_code = StageCode.TC
        self.tag_configs = tag_configs
        self._init_action_tags()

        self.gamma = cfg.gamma
        self.min_n_step = cfg.min_n_step
        self.max_n_step = cfg.max_n_step
        assert self.min_n_step > 0
        assert self.max_n_step >= self.min_n_step
        self.step_info = {
            n: NStepInfo(n) for n in range(self.min_n_step, self.max_n_step + 1)
        }

    def _init_action_tags(self):
        self.action_tags = []
        for tag_config in self.tag_configs:
            name = tag_config.name
            if tag_config.is_action:
                self.action_tags.append(name)

    def _reset_step_info(self):
        self.step_info = {
            n: NStepInfo(n) for n in range(self.min_n_step, self.max_n_step + 1)
        }

    def _make_steps(self, pf) -> list[Step]:
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
        if pf.data.empty:  # pretend like nothing happened but raise a warning...
            warnings.warn("Empty dataframe passed to transition creator", stacklevel=2)
            return pf

        tc_ts = pf.temporal_state.get(self.stage_code)
        assert isinstance(tc_ts, AllTheTimeTS | None)

        steps = self._make_steps(pf)
        transitions = []
        for step in steps:
            if has_nan(step):
                self._reset_step_info()  # nuke the step info
            else:
                transitions += self._update(step)

        if tc_ts is None:
            tc_ts = AllTheTimeTS(self.step_info)
        else:
            tc_ts.step_info = self.step_info

        pf.temporal_state[self.stage_code] = tc_ts
        pf.transitions = transitions

        return pf

    def _update(self, step: Step) -> list[NewTransition]:
        """
        Updates all the step queues, n_step_rewards, and n_step_gammas stored in self.step_info with the new step,
        then returns any produced transitions.
        """
        new_transitions = []
        for n in range(self.min_n_step, self.max_n_step + 1):
            n_step_info = self.step_info[n]
            step_q = n_step_info.step_q
            step_q.append(step)

            is_full = len(step_q) == n + 1
            if is_full:
                n_step_info.n_step_reward, n_step_info.n_step_gamma = update_n_step_reward_gamma(
                    n_step_info.n_step_reward,
                    n_step_info.n_step_gamma,
                    step_q,
                )

                assert n_step_info.n_step_reward is not None
                assert n_step_info.n_step_gamma is not None

                new_transition = NewTransition(
                    list(step_q),
                    n_step_reward=n_step_info.n_step_reward,
                    n_step_gamma=n_step_info.n_step_gamma,
                )

                new_transitions.append(new_transition)

        return new_transitions
