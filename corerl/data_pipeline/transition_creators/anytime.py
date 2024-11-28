import numpy as np
from collections import deque

from dataclasses import dataclass, field

from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, GORAS, NewTransition2
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
    n_step: None | int = None


@dataclass
class SAR:
    state: np.ndarray
    action: np.ndarray
    reward: float


@dataclass
class AnytimeTemporalState(TransitionCreatorTemporalState):
    aw_goras: list[GORAS] = field(default_factory=list)
    prev_data_gap: bool = False
    supress_aw_end: bool = False


class AnytimeTransitionCreator(BaseTransitionCreator):
    def __init__(self, cfg: AnytimeTransitionCreatorConfig):
        super().__init__(cfg)
        self.steps_per_decision = cfg.steps_per_decision
        self.gamma = cfg.gamma
        if cfg.n_step is None:
            self.queue_len = self.steps_per_decision
        else:
            self.queue_len = cfg.n_step

        self.prev_data_gap = False

    def _inner_call(self,
                    pf: PipelineFrame,
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[NewTransition2], AnytimeTemporalState]:

        assert isinstance(tc_ts, AnytimeTemporalState) or tc_ts is None
        actions = pf.data[pf.action_tags].to_numpy()
        observations = pf.data[pf.obs_tags].to_numpy()
        states = pf.data['state'].to_numpy()
        rewards = pf.data['reward'].to_numpy()
        assert len(actions) == len(states)

        aw_goras, last_action, transitions, supress_aw_end = self._restore_from_ts(actions, tc_ts)

        for i in range(len(actions)):
            action = actions[i]
            goras = GORAS(
                gamma=self.gamma,
                obs=observations[i],
                reward=rewards[i],
                action=action,
                state=states[i],
            )

            action_window_ends = last_action is not None and not np.allclose(action, last_action)
            if action_window_ends and not supress_aw_end:
                transitions += self._make_decision_window_transitions(aw_goras)
                aw_goras = [aw_goras[-1]]

            aw_goras.append(goras)

            decision_window_ends = len(aw_goras) == self.steps_per_decision + 1
            if decision_window_ends:
                transitions += self._make_decision_window_transitions(aw_goras)
                aw_goras = [aw_goras[-1]]
                supress_aw_end = True
            else:
                supress_aw_end = False

            last_action = action

        tc_ts = AnytimeTemporalState(aw_goras, pf.data_gap, supress_aw_end)

        return transitions, tc_ts

    def _restore_from_ts(
            self,
            actions: np.ndarray,
            tc_ts: AnytimeTemporalState | None) -> tuple[list[GORAS], np.ndarray | None, list[NewTransition2], bool]:
        """
        Restores the state of the transition creator from the temporal state (tc_ts).
        This temporal state is summarized in tc_ts.prev_data_gap and tc_ts.aw_sars.
        If there are SARs in tc_ts.aw_sars, then there were SARs that did not get processed in the last call of
        the transition creator. If there was not a datagap, we continue processing these SARs, so this function
        will return aw_sars. If the previously processed pipeframe had a datagap, then we need to add these transitions
        """

        reset_ts = tc_ts is None

        aw_sars = []
        transitions = []
        last_action = None
        supress_aw_end = False

        if reset_ts is None and len(actions):
            last_action = actions[0]

        elif not reset_ts:
            aw_sars = tc_ts.aw_goras
            supress_aw_end = tc_ts.supress_aw_end

            _check_actions_valid(aw_sars[1:])

            last_action = aw_sars[-1].action
            transitions = []

            if tc_ts.prev_data_gap:
                transitions = self._make_decision_window_transitions(aw_sars)
                supress_aw_end = False
                aw_sars = []

        return aw_sars, last_action, transitions, supress_aw_end

    def _make_decision_window_transitions(
            self, dw_goras: list[GORAS]) -> list[NewTransition2]:
        """
        Makes transitions for a decision window (dw), which starts and ends with a decision point.

        dw_sars contains a list of (s, a, r) tuples for that decision window.
        """
        _check_actions_valid(dw_goras)
        dw_transitions = []
        action = dw_goras[1].action

        goras_queue = deque([], self.queue_len)
        goras_queue.appendleft(dw_goras[-1])

        for step_backwards in range(len(dw_goras) - 2, -1, -1):
            pre_goras = dw_goras[step_backwards]
            n_step_reward = _get_n_step_reward(goras_queue, self.gamma)

            boot_goras = goras_queue[-1]
            post_goras = GORAS(
                gamma=boot_goras.gamma,
                obs=boot_goras.obs,
                reward=n_step_reward,
                action=boot_goras.action,
                state=boot_goras.state,
            )

            transition = NewTransition2(pre_goras, post_goras, len(goras_queue))
            goras_queue.appendleft(pre_goras)
            dw_transitions.append(transition)

        dw_transitions.reverse()

        return dw_transitions


def split_list_with_remainder(lst: list, n: int) -> list[list]:
    portions = [lst[i:i + n] for i in range(0, len(lst), n)]
    return portions


def _check_actions_valid(goras_list: list[GORAS]):
    if len(goras_list) < 2:
        return
    actions = [o.action for o in goras_list]
    first_action = actions[1]
    for action in actions[2:]:
        assert np.allclose(first_action, action), "All actions within a decision window must be equal."


def _get_n_step_reward(goras_queue: deque[GORAS], gamma: float) -> float:
    n_step_reward = 0
    steps_until_bootstrap = len(goras_queue)
    for step in range(steps_until_bootstrap):
        cumulant_step = goras_queue[step].reward
        n_step_reward += (gamma ** step) * cumulant_step
    return n_step_reward


transition_creator_group.dispatcher(AnytimeTransitionCreator)
