import numpy as np
import gymnasium
import time

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction


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
    Interaction that will repeat an action for some length of time, while the observation is still
    updated more frequently
    """

    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor: BaseStateConstructor):
        super().__init__(cfg, env, state_constructor)
        self.steps_per_decision = cfg.decision_steps  # how many observation steps per decision step
        self.obs_length = cfg.obs_length  # how often to update the observation
        self.gamma = cfg.gamma
        assert self.obs_length > 0, "Step length should be greater than 0."

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[list[tuple], list[dict]]:

        state_list = [state]
        reward_list = np.zeros(self.steps_per_decision)
        terminated_list = []
        truncate_list = []
        env_info_list = []

        for obs_step in range(self.steps_per_decision):
            obs_end_time = time.time() + self.obs_length
            denormalized_action = self.action_normalizer.denormalize(action)
            next_observation, raw_reward, terminated, _, env_info = self.env.step(denormalized_action)
            truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment

            reward = self.reward_normalizer(raw_reward)

            gamma_exponents = create_countdown_vector(obs_step, self.steps_per_decision)
            gammas = np.power(self.gamma, gamma_exponents)
            reward_list += gammas * reward

            is_decision_point = (obs_step == self.steps_per_decision - 1)
            is_done = is_decision_point or terminated or truncate
            if is_done:
                next_state = self.state_constructor(next_observation, decision_point=True)
            else:
                next_state = self.state_constructor(next_observation)

            state_list.append(next_state)
            terminated_list.append(terminated)
            truncate_list.append(truncate)
            env_info_list.append(env_info)

            time.sleep(obs_end_time-time.time())
            if terminated or truncate:
                break

        # assemble the transitions
        num_completed_steps = len(truncate_list)
        transitions = []
        for obs_step in range(num_completed_steps):  # note: we do not return a transition for the final state
            transition = (state_list[obs_step], action, reward_list[obs_step], next_state, terminated_list[obs_step],
                          truncate_list[obs_step])
            transitions.append(transition)

        return transitions, env_info_list
