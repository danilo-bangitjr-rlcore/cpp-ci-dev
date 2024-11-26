import numpy as np
from collections import deque

from dataclasses import dataclass, field

from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import NewTransition, PipelineFrame
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
    aw_sars: list[SAR] = field(default_factory=list)
    prev_data_gap: bool = False


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
            -> tuple[list[NewTransition], AnytimeTemporalState]:

        assert isinstance(tc_ts, AnytimeTemporalState) or tc_ts is None
        actions = pf.data[pf.action_tags].to_numpy()
        states = pf.data['state'].to_numpy()
        rewards = pf.data['reward'].to_numpy()
        assert len(actions) == len(states)

        aw_sars, last_action, transitions = self._restore_from_ts(actions, tc_ts)

        for i in range(len(actions)):
            state, action, reward = states[i], actions[i], rewards[i]
            sar = SAR(state, action, reward)

            if last_action is not None and (not np.allclose(action, last_action) or len(aw_sars) == self.steps_per_decision):
                transitions += self._make_decision_window_transitions(aw_sars)
                aw_sars = [sar]

            last_action = action
            aw_sars.append(sar)

        tc_ts = AnytimeTemporalState(aw_sars, pf.data_gap)

        return transitions, tc_ts

    def _restore_from_ts(
            self,
            actions: np.ndarray,
            tc_ts: AnytimeTemporalState | None) -> tuple[list[SAR], np.ndarray | None, list[NewTransition]]:
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

        if reset_ts is None and len(actions):
            last_action = actions[0]

        elif not reset_ts:
            aw_sars = tc_ts.aw_sars
            _check_actions_equal(aw_sars)
            last_action = aw_sars[-1].action
            transitions = []

            if tc_ts.prev_data_gap:
                transitions = self._make_decision_window_transitions(aw_sars)
                aw_sars = []

        return aw_sars, last_action, transitions

    def _make_decision_window_transitions(
            self, dw_sars: list[SAR]) -> list[NewTransition]:
        """
        Makes transitions for a decision window (dw), which starts and ends with a decision point.

        dw_sars contains a list of (s, a, r) tuples for that decision window.
        """

        next_state_queue = deque([], self.queue_len)
        reward_queue = deque([], self.queue_len)
        next_state = dw_sars[-1].state
        next_state_queue.append(next_state)
        dw_transitions = []

        dw_sars = dw_sars[:-1]
        dw_sars.reverse()

        for sar in dw_sars:
            state, action, reward = sar.state, sar.action, sar.reward
            reward_queue.appendleft(reward)
            n_step_reward = _get_n_step_reward(reward_queue, self.gamma)

            transition = NewTransition(
                state=tensor(state),
                action=tensor(action),
                n_steps=len(reward_queue),
                n_step_reward=n_step_reward,
                next_state=tensor(next_state_queue[-1]),
                terminated=False,
                truncate=False
            )

            next_state_queue.appendleft(state)
            dw_transitions.append(transition)

        dw_transitions.reverse()
        return dw_transitions


def split_list_with_remainder(lst: list, n: int) -> list[list]:
    portions = [lst[i:i + n] for i in range(0, len(lst), n)]
    return portions


def _check_actions_equal(sars: list[SAR]):
    actions = [o.action for o in sars]
    first_action = actions[0]
    for action in actions[1:]:
        assert first_action == action, "All actions within a decision window must be equal."


def _get_n_step_reward(reward_queue: deque[float], gamma: float) -> float:
    n_step_reward = 0
    steps_until_bootstrap = len(reward_queue)
    for step in range(steps_until_bootstrap):
        cumulant_step = reward_queue[step]
        n_step_reward += (gamma ** step) * cumulant_step
    return n_step_reward


transition_creator_group.dispatcher(AnytimeTransitionCreator)
