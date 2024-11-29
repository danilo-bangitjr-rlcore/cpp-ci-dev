import numpy as np
import torch
from collections import deque

from dataclasses import dataclass, field

from corerl.component.network.utils import tensor
from corerl.data_pipeline.datatypes import PipelineFrame, GORAS, NewTransition
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


@dataclass
class AnytimeTemporalState(TransitionCreatorTemporalState):
    dw_goras: list[GORAS] = field(default_factory=list)  # the goras from the LAST decision window of the previous pf
    prev_data_gap: bool = False


def _get_tags(pf, tags: list[str] | str) -> torch.Tensor:
    return tensor(pf.data[tags].to_numpy())


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
            -> tuple[list[NewTransition], AnytimeTemporalState | None]:

        assert isinstance(tc_ts, AnytimeTemporalState) or tc_ts is None

        actions = _get_tags(pf, pf.action_tags)
        observations = _get_tags(pf, pf.obs_tags)
        states = _get_tags(pf, pf.state_tags)
        rewards = pf.data['reward'].to_numpy()

        if not len(actions):
            return [], tc_ts

        assert len(actions) == len(states)

        dw_goras, transitions = self._restore_from_ts(actions, tc_ts)

        for i in range(len(actions)):
            action = actions[i]
            goras = GORAS(
                gamma=self.gamma,
                obs=observations[i],
                reward=rewards[i],
                action=action,
                state=states[i],
            )
            dw_goras.append(goras)

            next_action = actions[i + 1] if i != len(actions) - 1 else action
            action_change = not torch.allclose(action, next_action)
            reached_n_step = len(dw_goras) == self.queue_len + 1 if self.queue_len is not None else False
            if reached_n_step or action_change:
                transitions += self._make_decision_window_transitions(dw_goras)
                dw_goras = [dw_goras[-1]]

        tc_ts = AnytimeTemporalState(dw_goras, pf.data_gap)

        return transitions, tc_ts

    def _restore_from_ts(
            self,
            actions: torch.Tensor,
            tc_ts: AnytimeTemporalState | None) -> tuple[list[GORAS], list[NewTransition]]:
        """
        Restores the state of the transition creator from the temporal state (tc_ts).
        This temporal state is summarized in tc_ts.prev_data_gap and tc_ts.d_goras and supress_aw_end/
        If there are GORASs in tc_ts.dw_goras, then there were GORASs that did not get processed in the last call of
        the transition creator. If there was not a datagap, we continue processing these GORASs, so this function
        will return dw_goras. If the previously processed pipeframe had a datagap,
        then we need to add these transitions.
        """

        # Case 1: tc_ts is None
        if tc_ts is None:
            return [], []

        # Case 2: Valid tc_ts exists
        dw_goras = tc_ts.dw_goras
        if not len(dw_goras):
            return [], []

        first_action = actions[0]
        last_pf_action = dw_goras[-1].action

        if tc_ts.prev_data_gap:
            transitions = self._make_decision_window_transitions(dw_goras)
            dw_goras = []
        elif not torch.allclose(first_action, last_pf_action):
            transitions = self._make_decision_window_transitions(dw_goras)
            dw_goras = [dw_goras[-1]]
        else:
            transitions = []

        return dw_goras, transitions

    def _make_decision_window_transitions(
            self, dw_goras: list[GORAS]) -> list[NewTransition]:
        """
        Makes transitions for a decision window (dw), which starts and ends with a decision point.
        dw_goras contains a list of GORAS for that decision window.
        NOTE that the first GORAS may have a different action than the rest.
        """
        _check_actions_valid(dw_goras)
        dw_transitions = []
        goras_queue = deque([], self.queue_len)
        goras_queue.appendleft(dw_goras[-1])

        for step_backwards in range(len(dw_goras) - 2, -1, -1):
            pre_goras = dw_goras[step_backwards]
            n_step_reward, n_step_gamma = _get_n_step_reward_gamma(goras_queue)

            boot_goras = goras_queue[-1]
            n_steps = len(goras_queue)

            post_goras = GORAS(
                gamma=boot_goras.gamma ** n_steps,
                obs=boot_goras.obs,
                reward=n_step_reward,
                action=boot_goras.action,
                state=boot_goras.state,
            )

            transition = NewTransition(pre_goras, post_goras, n_steps)
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


def _get_n_step_reward_gamma(goras_queue: deque[GORAS]) -> tuple[float, float]:
    """
    Gets discounted sum of rewards in the goras_queue and also the gamma for
    bootstrapping off the final state in the queue
    """

    n_step_reward = 0
    n_step_gamma = 1
    steps_until_bootstrap = len(goras_queue)

    for step in range(0, steps_until_bootstrap):
        reward = goras_queue[step].reward
        gamma = goras_queue[step].gamma
        n_step_reward += n_step_gamma * reward
        n_step_gamma *= gamma

    return n_step_reward, n_step_gamma


transition_creator_group.dispatcher(AnytimeTransitionCreator)
