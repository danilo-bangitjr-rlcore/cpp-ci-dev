import numpy as np
import gymnasium
import time
from copy import deepcopy
from collections import deque

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data import Transition


def create_countdown_vector(start, total_length):
    # Create the countdown part
    assert start + 1 <= total_length
    countdown_length = start + 1
    countdown = np.arange(start, -1, -1)[:countdown_length]

    # Determine the length of zero padding needed
    zero_padding_length = total_length - countdown_length

    # Create the zero padding
    zero_padding = np.zeros(zero_padding_length, dtype=int)

    # Concatenate countdown and zero padding
    result = np.concatenate([countdown, zero_padding])
    return result


class AnytimeInteraction(NormalizerInteraction):
    """
    Interaction that will repeat an action for some length of time, while the
    observation is still updated more frequently
    """

    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert
    ):
        super().__init__(cfg, env, state_constructor, alerts)
        self.n_step = cfg.n_step
        self.warmup_steps = cfg.warmup_steps

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[dict]]: # TODO: update this
        partial_transitions = []  # contains (O, S, A, R, S_DP, O', S', C)
        alert_info_list = []
        env_info_list = []

        denormalized_action = self.action_normalizer.denormalize(action)

        trunc = False
        term = False
        prev_decision_point = True
        for obs_step in range(self.steps_per_decision):
            out = self.env.step(denormalized_action)  # env.step() already ensures self.obs_length has elapsed
            next_obs, raw_reward, term, env_trunc, env_info = out
            env_info_list.append(env_info)

            trunc = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
            reward = self.reward_normalizer(raw_reward)
            next_obs = self.obs_normalizer(next_obs)

            curr_cumulants = self.get_cumulants(reward, next_obs)  # for alerts

            alert_info = self.get_step_alerts(denormalized_action, action, self.last_state, next_obs, reward)
            alert_info_list.append(alert_info)

            decision_point = (obs_step == self.steps_per_decision - 1)
            next_state = self.state_constructor(next_obs, action, decision_point=decision_point)

            partial_transitions.append((self.last_obs,
                                        self.last_state,
                                        action,
                                        reward,
                                        prev_decision_point,
                                        next_obs,
                                        next_state,
                                        curr_cumulants))

            prev_decision_point = decision_point
            self.last_state = next_state
            self.last_obs = next_obs

            if term or trunc:
                break

        # Create transitions
        new_agent_transitions, new_alert_transitions = self.create_n_step_transitions(partial_transitions,
                                                                                      self.last_state,
                                                                                      self.last_obs,
                                                                                      term,
                                                                                      trunc)

        # Only train on transitions where there weren't any alerts
        agent_train_transitions = self.get_agent_train_transitions(new_agent_transitions, alert_info_list)
        alert_train_transitions = self.get_alert_train_transitions(new_alert_transitions, alert_info_list)

        return new_agent_transitions, agent_train_transitions, alert_train_transitions, alert_info_list, env_info_list

    def update_n_step_cumulants(self, n_step_cumulant_q, new_cumulant, gammas) -> np.ndarray:
        """
        Recursively updating n-step cumulant
        """
        num_cumulants = len(new_cumulant)
        n_step_cumulant_q.appendleft([0.0 for _ in range(num_cumulants)])
        np_n_step_cumulants = np.array(n_step_cumulant_q)
        np_new_cumulant = np.array([new_cumulant for _ in range(len(n_step_cumulant_q))])
        np_n_step_cumulants = np_new_cumulant + (gammas * np_n_step_cumulants)

        return np_n_step_cumulants

    def create_n_step_transitions(self,
                                  partial_transitions: list[tuple],
                                  boot_state: np.ndarray,
                                  boot_obs: np.ndarray,
                                  term: bool,
                                  trunc: bool) -> (list[tuple], list[tuple]):
        # If n_step = 0, create transitions where all states bootstrap off the state at the next decision point
        # If n_step > 0, create transitions where states bootstrap off the state n steps into the future.
        # If the state n steps ahead is beyond the next decision point, bootstrap off the state at the decision point
        agent_transitions = []
        alert_transitions = []

        alert_gammas = np.array(self.alerts.get_discount_factors())

        dp_counter = 1
        # Queues used to track the n-step bootstrapping
        if self.n_step == 0 or self.n_step >= self.steps_per_decision:
            n_step_rewards = deque([], self.steps_per_decision)
            n_step_cumulants = deque([], self.steps_per_decision)
            boot_state_queue = deque([], self.steps_per_decision)
            boot_obs_queue = deque([], self.steps_per_decision)
            term_queue = deque([], self.steps_per_decision)
            trunc_queue = deque([], self.steps_per_decision)
        else:
            n_step_rewards = deque([], self.n_step)
            n_step_cumulants = deque([], self.n_step)
            boot_state_queue = deque([], self.n_step)
            boot_obs_queue = deque([], self.n_step)
            term_queue = deque([], self.n_step)
            trunc_queue = deque([], self.n_step)

        boot_state_queue.appendleft(boot_state)
        boot_obs_queue.appendleft(boot_obs)
        term_queue.appendleft(term)
        trunc_queue.appendleft(trunc)

        # Iteratively create the transitions
        for i in range(len(partial_transitions) - 1, -1, -1):
            obs = partial_transitions[i][0]
            state = partial_transitions[i][1]
            action = partial_transitions[i][2]
            reward = partial_transitions[i][3]
            s_dp = partial_transitions[i][4]
            next_obs = partial_transitions[i][5]
            next_state = partial_transitions[i][6]
            cumulants = partial_transitions[i][7]

            # Create agent transition
            np_n_step_rewards = self.update_n_step_cumulants(n_step_rewards, np.array([reward]), self.gamma)

            # Shared amongst agent and alert transitions
            gamma_exp = len(np_n_step_rewards)
            ns_dp = dp_counter <= boot_state_queue.maxlen

            agent_transition = Transition(
                obs,
                state,
                action,
                next_obs,  # the immediate next obs
                next_state,  # the immediate next state
                np_n_step_rewards[-1].item(),
                boot_obs_queue[-1],  # the obs we bootstrap off
                boot_state_queue[-1],  # the state we bootstrap off
                term_queue[-1],
                trunc_queue[-1],
                s_dp,
                ns_dp,
                gamma_exp)

            agent_transitions.append(agent_transition)

            # Create alert transition(s)
            np_n_step_cumulants = self.update_n_step_cumulants(n_step_cumulants, cumulants, alert_gammas)

            step_alert_transitions = []
            alert_start_ind = 0
            for alert in self.alerts.alerts:
                alert_end_ind = alert_start_ind + alert.get_dim()

                alert_transition = Transition(
                    obs,
                    state,
                    action,
                    next_obs,  # the immediate next obs
                    next_state,  # the immediate next state
                    np_n_step_cumulants[-1][alert_start_ind: alert_end_ind].item(),
                    boot_obs_queue[-1],  # the obs we bootstrap off
                    boot_state_queue[-1],  # the state we bootstrap off
                    term_queue[-1],
                    trunc_queue[-1],
                    s_dp,
                    ns_dp,
                    gamma_exp)

                step_alert_transitions.append(alert_transition)
                alert_start_ind = alert_end_ind

            alert_transitions.append(step_alert_transitions)

            # Update queues and counters
            dp_counter += 1
            boot_state_queue.appendleft(state)
            boot_obs_queue.appendleft(obs)
            term_queue.appendleft(
                False)  # Can assume the partial transitions before the last one don't truncate or terminate
            trunc_queue.appendleft(False)
            n_step_rewards = deque(np_n_step_rewards, n_step_rewards.maxlen)
            n_step_cumulants = deque(np_n_step_cumulants, n_step_cumulants.maxlen)

        agent_transitions.reverse()
        alert_transitions.reverse()

        return agent_transitions, alert_transitions
