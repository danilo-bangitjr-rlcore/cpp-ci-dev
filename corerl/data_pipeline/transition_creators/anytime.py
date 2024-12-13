import numpy as np
import torch
import pandas as pd

from typing import Literal, Protocol
from copy import copy
from dataclasses import dataclass

from corerl.utils.device import device
from corerl.configs.config import config, MISSING, interpolate
from corerl.data_pipeline.tag_config import TagConfig
from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, Step, NewTransition
from corerl.data_pipeline.transition_creators.base import (
    BaseTransitionCreator,
    BaseTransitionCreatorConfig,
    transition_creator_group,
    TransitionCreatorTemporalState,
)


@config(frozen=True)
class AnytimeTransitionCreatorConfig(BaseTransitionCreatorConfig):
    name: Literal['anytime'] = "anytime"
    steps_per_decision: int = MISSING
    gamma: float = interpolate('${experiment.gamma}')
    n_step: None | int = None  # if n_step is None, will bootstrap off of the next decision point
    only_dp_transitions: bool = False  # whether we only want to return transitions between decision points
    countdown: str = 'one_hot'


@dataclass
class StepInfo:
    step: Step
    sud: int


@dataclass
class AnytimeTemporalState(TransitionCreatorTemporalState):
    prev_step_info_list: list[StepInfo]  # left over step from the last time we made transitions
    prev_data_gap: bool


class AnytimeTransitionCreator(BaseTransitionCreator):
    def __init__(
            self,
            cfg: AnytimeTransitionCreatorConfig,
            tag_configs: list[TagConfig],
    ):
        super().__init__(cfg, tag_configs)
        self.steps_per_decision = cfg.steps_per_decision
        self.gamma = cfg.gamma
        if cfg.n_step is None:
            self.max_boot_len = self.steps_per_decision
        else:
            self.max_boot_len = cfg.n_step

        self.only_dp_transitions = cfg.only_dp_transitions
        self.countdown_adder = init_countdown_adder(cfg.countdown)

    def _inner_call(self,
                    pf: PipelineFrame,
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[NewTransition], TransitionCreatorTemporalState | None]:

        assert isinstance(tc_ts, AnytimeTemporalState | None)

        transitions = []

        result = _split_at_nans(pf.data)
        # if the entire dataframe was nan, we have now detected a datagap
        if not len(result):
            if isinstance(tc_ts, AnytimeTemporalState):
                tc_ts.prev_data_gap = True

        for df, post_df_data_gap in _split_at_nans(pf.data):
            new_transitions, tc_ts = self._process_df(df, tc_ts, post_df_data_gap)
            transitions += new_transitions

        return transitions, tc_ts

    def _process_df(self,
                    df: pd.DataFrame,
                    tc_ts: AnytimeTemporalState | None,
                    post_df_data_gap: bool) \
            -> tuple[list[NewTransition], AnytimeTemporalState | None]:
        """
        Produces transitions for a df without data gaps. post_df_data_gap tells us whether a datagap
        follows this df. This is needed for producing the temporal state for the NEXT df.
        """

        actions = _get_tags(df, self.action_tags)
        state_tags = sorted(
            set(df.columns) - set(self.action_tags) - {'reward', 'trunc', 'term'}
        )
        states = _get_tags(df, state_tags)
        rewards = df['reward'].to_numpy()

        gammas = np.ones(len(rewards))
        if 'term' in df.columns:
            gammas = 1 - df['term'].to_numpy()

        if not len(actions):
            return [], tc_ts

        assert len(actions) == len(states) and len(states) == len(rewards)

        step_info_list, transitions, steps_until_decision = self._restore_from_ts(actions, tc_ts)
        for i in range(len(actions)):
            action = actions[i]
            step = Step(
                reward=rewards[i],
                action=action,
                gamma=self.gamma * gammas[i],
                state=states[i],
                dp=False,
            )

            si = StepInfo(
                step,
                steps_until_decision
            )
            step_info_list.append(si)

            next_action = actions[i + 1] if i != len(actions) - 1 else action
            action_change = not torch.allclose(action, next_action)
            reached_n_step = len(step_info_list) == self.max_boot_len + 1
            if reached_n_step or action_change:
                transitions += self._make_transitions(step_info_list)
                """
                If the action changes, we will restart the countdown in update_countdown.
                But we also need to make sue that si.sud is consistent with this updated countdown.
                """
                if action_change:
                    si.sud = self.steps_per_decision

                step_info_list = [si]

            steps_until_decision = update_sud(
                steps_until_decision,
                self.steps_per_decision,
                action_change,
            )
        tc_ts = AnytimeTemporalState(
            prev_step_info_list=step_info_list,
            prev_data_gap=post_df_data_gap,
        )

        return transitions, tc_ts

    def _restore_from_ts(
            self,
            actions: torch.Tensor,
            tc_ts: AnytimeTemporalState | None) -> tuple[list[StepInfo], list[NewTransition], int]:
        """
        Restores the state of the transition creator from the temporal state (tc_ts).
        This temporal state is summarized in tc_ts.prev_data_gap and tc_ts.step_info_list
        If there are steps in tc_ts.step_list, then there were steps that did not get processed in the last call of
        the transition creator. If there was not a datagap, we continue processing these steps, so this function
        will return step_list. If the previously processed pipeframe had a datagap,
        then we need to also return these transitions.

        Returns:
                step_info_list : unprocessed list of step_infos
                transitions: transitions (only produced if there was a datagap)
                steps_until_decision: the steps_until_decision counter for the NEXT step
        """

        # Case 1: tc_ts is None
        if tc_ts is None:
            return [], [], self.steps_per_decision

        # Case 2: Valid tc_ts exists but there were no transitions in it.
        step_info_list = tc_ts.prev_step_info_list
        if not len(step_info_list):
            return [], [], self.steps_per_decision

        first_action = actions[0]  # first action of the next df
        last_action = step_info_list[-1].step.action  # last action of the prev df
        action_change = not torch.allclose(last_action, first_action)

        if tc_ts.prev_data_gap:
            transitions = self._make_transitions(step_info_list)
            step_info_list = []
            steps_until_decision = self.steps_per_decision

        elif action_change:
            transitions = self._make_transitions(step_info_list)
            last_si = step_info_list[-1]
            last_si.sud = self.steps_per_decision
            step_info_list = [last_si]
            steps_until_decision = self.steps_per_decision - 1  # minus one is because this is for the NEXT step

        else:  # no data gap or action change, just continue processing steps
            transitions = []
            steps_until_decision = step_info_list[-1].sud - 1  # minus one is because this is for the NEXT step

        return step_info_list, transitions, steps_until_decision

    def _make_transitions(
            self, step_info_list: list[StepInfo]) -> list[NewTransition]:
        """
        Makes transitions for a list of StepInfo objects.
        """
        _check_actions_valid(step_info_list)
        transitions = []

        assert len(step_info_list) <= self.max_boot_len + 1

        boot_step = self.countdown_adder(
            step_info_list[-1],
            steps_per_decision=self.steps_per_decision)

        # boot_sud = step_info_list[-1].sud
        # boot_dp = boot_sud == self.steps_per_decision or boot_sud == 0

        n_step_reward = boot_step.reward
        n_step_gamma = boot_step.gamma
        # last_step_idx = len(step_info_list) - 1
        for step_backwards in range(len(step_info_list) - 2, -1, -1):
            step_info = step_info_list[step_backwards]
            prior_step = self.countdown_adder(step_info, steps_per_decision=self.steps_per_decision)
            prior_step.dp = step_info.sud == self.steps_per_decision
            """
            if only_dp_transitions is False, then make_transitions is always True
            if only_dp_transitions is True, then make_transitions False except for the final transition
            """
            make_transition = not self.only_dp_transitions or step_backwards == 0
            if make_transition:
                transition = NewTransition(
                    steps=[prior_step, boot_step],
                    n_step_reward=n_step_reward,
                    n_step_gamma=n_step_gamma,
                    )
                transitions.append(transition)

            n_step_reward = prior_step.reward + boot_step.gamma * n_step_reward
            n_step_gamma *= boot_step.gamma

        transitions.reverse()

        return transitions


def _get_tags(df: pd.DataFrame, tags: list[str] | str) -> torch.Tensor:
    return tensor(df[tags].to_numpy())


def _split_at_nans(df: pd.DataFrame) -> list[tuple[pd.DataFrame, bool]]:
    """
    Splits a df where any of the columns have nans and prunes out these nan rows.
    Returns a list of tuples where the first entry of a tuple is the df and the second is whether there was a
    data gap (nan row) after this df.
    """

    nan_rows = df.isna().any(axis=1)
    split_indices = df[nan_rows].index.tolist()

    if not split_indices:
        return [(df, False)]

    result = []

    if split_indices[0] > df.index[0]:
        first_chunk = df.loc[:split_indices[0]].iloc[:-1]
        result.append((first_chunk, True))

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        chunk = df.loc[start:end].iloc[1:-1]
        result.append((chunk, True))

    if split_indices[-1] < df.index[-1]:
        last_chunk = df.loc[split_indices[-1]:].iloc[1:]
        result.append((last_chunk, False))

    return [(sub_df, has_nan) for sub_df, has_nan in result if not sub_df.empty]


def _check_actions_valid(step_info_list: list[StepInfo]) -> None:
    if len(step_info_list) < 2:
        return
    actions = [si.step.action for si in step_info_list]
    first_action = actions[1]
    for action in actions[2:]:
        assert np.allclose(first_action, action), "All actions within a decision window must be equal."


def update_sud(
        steps_until_decision: int,
        steps_per_decision: int,
        action_change) -> int:
    if action_change:
        return steps_per_decision - 1
    elif steps_until_decision == 0:
        return steps_per_decision
    return steps_until_decision - 1


transition_creator_group.dispatcher(AnytimeTransitionCreator)


class CountDownAdder(Protocol):
    def __call__(self, step: Step, steps_until_decision: int, steps_per_decision: int) -> Step:
        ...


class CountDownCaller(Protocol):
    def __call__(self, step_info: StepInfo, steps_per_decision: int) -> Step:
        ...


def init_countdown_adder(name: str) -> CountDownCaller:
    f = None
    match name:
        case 'null':
            f = add_null_countdown
        case 'one_hot':
            f = add_one_hot_countdown
        case 'float':
            f = add_float_countdown
        case 'int':
            f = add_int_countdown
        case _:
            raise NotImplementedError

    return lambda step_info, steps_per_decision: count_down_adder(step_info, steps_per_decision, f)


def count_down_adder(step_info: StepInfo, steps_per_decision: int, func: CountDownAdder) -> Step:
    step = copy(step_info.step)

    return func(
        step,
        step_info.sud,
        steps_per_decision,
    )


def add_null_countdown(
        step: Step,
        steps_until_decision: int,
        steps_per_decision: int) -> Step:
    assert steps_until_decision <= steps_per_decision
    return step


def add_one_hot_countdown(
        step: Step,
        steps_until_decision: int,
        steps_per_decision: int) -> Step:
    assert steps_until_decision <= steps_per_decision
    one_hot = torch.zeros((steps_per_decision,), device=device.device)
    one_hot[steps_until_decision - 1] = 1
    step.state = torch.cat([step.state, one_hot])
    return step


def add_float_countdown(
        step: Step,
        steps_until_decision: int,
        steps_per_decision: int) -> Step:
    assert steps_until_decision <= steps_per_decision
    if steps_until_decision == 0:
        steps_until_decision = steps_per_decision
    countdown = tensor(np.array([steps_until_decision / steps_per_decision]))
    step.state = torch.cat([step.state, countdown])
    return step


def add_int_countdown(
        step: Step,
        steps_until_decision: int,
        steps_per_decision: int) -> Step:
    assert steps_until_decision <= steps_per_decision
    if steps_until_decision == 0:
        steps_until_decision = steps_per_decision
    countdown = tensor(np.array([steps_until_decision]))
    step.state = torch.cat([step.state, countdown])
    return step
