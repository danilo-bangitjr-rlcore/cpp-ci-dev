from typing import Optional
from copy import deepcopy
import gymnasium

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import Transition
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer


class OfflineAnytimeInteraction(AnytimeInteraction):
    """
    Interact with a continuous offline dataset (no data gaps) as if it were encountered online
    'transitions' is a list of consecutive transitions that only contain warmed up states
    """

    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert,
            transition_creator: AnytimeTransitionCreator,
            normalizer: ObsTransitionNormalizer,
            transitions: list[Transition]
    ):
        super().__init__(cfg, env, state_constructor, alerts, transition_creator, normalizer)

        # 'transitions' is produced by make_offline_transitions() in transition_creator.py
        # make_offline_transitions() already takes into account state constructor warmup in the transitions it returns
        self.offline_transitions = transitions
        assert len(self.offline_transitions) > 0

        # Keep track of which transition we're at in self.offline_transitions
        self.transition_ind = 0
        self.curr_decision_transitions = []
        self.alert_info_list = []

    def step(self) -> tuple[Optional[list[Transition]], Optional[list[Transition]], Optional[list[Transition]], dict, dict]:
        """
        Process the offline data, as if it were encountered online, until the next decision point is reached.
        Returns:
        - new_transitions: List of all produced transitions since the last decision point
        - train_transitions: The transitions within 'new_transitions' that didn't trigger an alert
        - alert_info: Dictionary describing which types of alerts were/weren't triggered
        - env_info: Dictionary describing env info
        """
        if self.transition_ind >= len(self.offline_transitions):
            return None, None, {}, {}
        else:
            transition = self.offline_transitions[self.transition_ind]
            self.curr_decision_transitions.append(transition)

            action = transition.action
            state = transition.state
            next_obs = transition.next_obs
            reward = transition.reward

            self.reward_sum += reward

            alert_info = self.get_step_alerts(action, state, next_obs, reward)
            self.alert_info_list.append(alert_info)

            # Filter transitions with alerts
            transitions, train_transitions = [], []  # NOTE: these lists may sometimes be empty if we are not at a decision point
            agent_train_transitions = []
            if transition.next_state_dp:
                # Only train on transitions where there weren't any alerts
                transitions = self.curr_decision_transitions
                train_transitions = self.get_train_transitions(transitions, self.alert_info_list)

                # if we are only returning dp transitions, we need to update the transition we are returning
                # we take the first transition, since it starts at the last decision point and goes to the next
                if self.only_dp_transitions:
                    # the first transition could have been filtered out by self.get_train_transitions(), so
                    # transitions[0].state_dp checks if it was that original first transition returned by
                    # self.transition_creator.make_online_transitions()
                    if train_transitions[0].state_dp:
                        transition = deepcopy(train_transitions[0])
                        transition.gamma_exponent = 1
                        transition.next_obs = transition.boot_obs
                        transition.next_state = transition.boot_state
                        transition.reward = self.reward_sum / self.steps_per_decision
                        agent_train_transitions = [transition]
                else:
                    agent_train_transitions = train_transitions
                self.reward_sum = 0
                self.curr_decision_transitions = []
                self.alert_info_list = []

            self.transition_ind += 1

            alert_train_transitions = train_transitions
            return transitions, agent_train_transitions, alert_train_transitions, alert_info, {}
