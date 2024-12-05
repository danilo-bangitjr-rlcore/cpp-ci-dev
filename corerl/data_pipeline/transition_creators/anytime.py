import numpy as np
import torch
import pandas as pd

from dataclasses import dataclass, field

from corerl.data_pipeline.tag_config import TagConfig
from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, RAGS, NewTransition
from corerl.utils.hydra import interpolate
from corerl.data_pipeline.transition_creators.base import (
    BaseTransitionCreator,
    BaseTransitionCreatorConfig,
    transition_creator_group,
    TransitionCreatorTemporalState,
)


@dataclass
class AnytimeTransitionCreatorConfig(BaseTransitionCreatorConfig):
    name: str = "anytime"
    steps_per_decision: int = interpolate('${interaction.steps_per_decision}')
    gamma: float = interpolate('${experiment.gamma}')
    n_step: None | int = None  # if n_step is None, will bootstrap off of the next decision point
    only_dp_transitions: bool = False  # whether we only want to return transitions between decision points


@dataclass
class AnytimeTemporalState(TransitionCreatorTemporalState):
    rags_list: list[RAGS] = field(default_factory=list)  # left over rags from the last time we made transitions
    prev_data_gap: bool = False


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

    def _inner_call(self,
                    pf: PipelineFrame,
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[NewTransition], TransitionCreatorTemporalState | None]:

        assert isinstance(tc_ts, AnytimeTemporalState | None)

        transitions = []
        for df, post_df_data_gap in _split_at_nans(pf.data):
            new_transitions, tc_ts = self._process_df(df, pf, tc_ts, post_df_data_gap)
            transitions += new_transitions

        return transitions, tc_ts

    def _process_df(self,
                    df: pd.DataFrame,
                    pf: PipelineFrame,
                    tc_ts: AnytimeTemporalState | None,
                    post_df_data_gap: bool) \
            -> tuple[list[NewTransition], AnytimeTemporalState | None]:
        """
        Produces transitions for a dataframe without data gaps. post_df_data_gap tells us whether a datagap
        follows this df. This is needed for producing the temporal state for the NEXT df

        """

        actions = _get_tags(df, self.action_tags)
        state_tags = sorted(
            set(pf.data.columns) - set(self.action_tags + ['reward'])
        )
        states = _get_tags(df, state_tags)
        rewards = pf.data['reward'].to_numpy()

        if not len(actions):
            return [], tc_ts

        assert len(actions) == len(states)

        rags_list, transitions = self._restore_from_ts(actions, tc_ts)

        for i in range(len(actions)):
            action = actions[i]
            rags = RAGS(
                reward=rewards[i],
                action=action,
                gamma=self.gamma,
                state=states[i],
            )
            rags_list.append(rags)

            next_action = actions[i + 1] if i != len(actions) - 1 else action
            action_change = not torch.allclose(action, next_action)
            reached_n_step = len(rags_list) == self.max_boot_len + 1
            if reached_n_step or action_change:
                transitions += self._make_transitions(rags_list)
                rags_list = [rags_list[-1]]

        tc_ts = AnytimeTemporalState(rags_list, post_df_data_gap)

        return transitions, tc_ts

    def _restore_from_ts(
            self,
            actions: torch.Tensor,
            tc_ts: AnytimeTemporalState | None) -> tuple[list[RAGS], list[NewTransition]]:
        """
        Restores the state of the transition creator from the temporal state (tc_ts).
        This temporal state is summarized in tc_ts.prev_data_gap and tc_ts.rags_list
        If there are RAGSs in tc_ts.rags_list, then there were RAGSs that did not get processed in the last call of
        the transition creator. If there was not a datagap, we continue processing these RAGSs, so this function
        will return rags_list. If the previously processed pipeframe had a datagap,
        then we need to add these transitions.
        """

        # Case 1: tc_ts is None
        if tc_ts is None:
            return [], []

        # Case 2: Valid tc_ts exists
        rags_list = tc_ts.rags_list
        if not len(rags_list):
            return [], []

        first_action = actions[0]
        last_pf_action = rags_list[-1].action

        if tc_ts.prev_data_gap:
            transitions = self._make_transitions(rags_list)
            rags_list = []
        elif not torch.allclose(first_action, last_pf_action):
            transitions = self._make_transitions(rags_list)
            rags_list = [rags_list[-1]]
        else:
            transitions = []

        return rags_list, transitions

    def _make_transitions(
            self, rags_list: list[RAGS]) -> list[NewTransition]:
        """
        Makes transitions for a list of RAGS.
        NOTE that the first RAGS may have a different action than the rest.
        """
        _check_actions_valid(rags_list)
        transitions = []

        assert len(rags_list) <= self.max_boot_len + 1

        boot_rags = rags_list[-1]
        n_step_reward = boot_rags.reward
        n_step_gamma = boot_rags.gamma
        last_rags_idx = len(rags_list) - 1

        for step_backwards in range(len(rags_list) - 2, -1, -1):
            prior_rags = rags_list[step_backwards]
            """
            if only_dp_transitions is False, then make_transitions is always True
            if only_dp_transitions is True, then make_transitions False except for the final transition
            """
            make_transition = not self.only_dp_transitions or step_backwards == 0
            if make_transition:
                n_steps = last_rags_idx - step_backwards
                post_rags = RAGS(
                    reward=n_step_reward,
                    action=boot_rags.action,
                    gamma=n_step_gamma,
                    state=boot_rags.state,
                )
                transition = NewTransition(prior_rags, post_rags, n_steps)
                transitions.append(transition)

            n_step_reward = prior_rags.reward + boot_rags.gamma * n_step_reward
            n_step_gamma *= boot_rags.gamma

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

    return [(subdf, has_nan) for subdf, has_nan in result if not subdf.empty]


def _check_actions_valid(rags_list: list[RAGS]):
    if len(rags_list) < 2:
        return
    actions = [o.action for o in rags_list]
    first_action = actions[1]
    for action in actions[2:]:
        assert np.allclose(first_action, action), "All actions within a decision window must be equal."


transition_creator_group.dispatcher(AnytimeTransitionCreator)
