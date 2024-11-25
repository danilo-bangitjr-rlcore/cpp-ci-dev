from idlelib.window import add_windows_to_menu

import numpy as np

from dataclasses import dataclass

from corerl.data_pipeline.datatypes import Transition, PipelineFrame
from corerl.data_pipeline.transition_creators.base import (
    BaseTransitionCreator,
    BaseTransitionCreatorConfig,
    transition_creator_group,
    TransitionCreatorTemporalState,
)


@dataclass
class AnytimeTransitionCreatorConfig(BaseTransitionCreatorConfig):
    name: str = "identity"
    steps_per_decision = 10  # TODO: fill this in


@dataclass
class AnytimeTemporalState(TransitionCreatorTemporalState):
    pass


@dataclass
class SAR:
    obs: np.ndarray
    action: np.ndarray
    reward: float


class AnytimeTransitionCreator(BaseTransitionCreator):
    def __init__(self, cfg: AnytimeTransitionCreatorConfig):
        super().__init__(cfg)
        self.steps_per_decision = cfg.steps_per_decision

    def _inner_call(self,
                    pf: PipelineFrame,
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[Transition], TransitionCreatorTemporalState]:

        actions = pf.data[pf.action_tags].to_numpy()
        states = pf.data['state'].to_numpy()
        # TODO: worry about reward

        assert len(actions) == len(states)
        aw_sars = []  # TODO: restore from the temporal state
        last_action = None  # TODO: restore from the temporal state
        for i in range(len(actions)):
            curr_state, curr_action = states[i], actions[i]
            sar = SAR(curr_state, curr_action, 0)
            if curr_action != last_action:
                transitions = self._make_action_window_transitions(aw_sars, curr_state)

            aw_sars.append(sar)

        # tc_ts = TransitionCreatorTemporalState()
        return [], tc_ts

    def _make_action_window_transitions(self,
                                        aw_sars: list[SAR],
                                        final_boot_state: np.ndarray) -> list[Transition]:

        decision_windows = split_list_with_remainder(aw_sars, self.steps_per_decision)
        num_dws = len(decision_windows)
        transitions = []
        for dw_idx, dw_sars in enumerate(decision_windows):
            is_last_dw = dw_idx == num_dws - 1

            if is_last_dw:
                boot_state = final_boot_state
            else:
                first_obs_of_next_dw = decision_windows[dw_idx + 1][0].obs
                boot_state = first_obs_of_next_dw

            transitions += self._make_decision_window_transitions(dw_sars, boot_state)

        return transitions

    def _make_decision_window_transitions(self, aw_sars, boot_state):
        pass


def split_list_with_remainder(lst: list, n: int) -> list[list]:
    portions = [lst[i:i + n] for i in range(0, len(lst), n)]
    return portions


transition_creator_group.dispatcher(AnytimeTransitionCreator)
