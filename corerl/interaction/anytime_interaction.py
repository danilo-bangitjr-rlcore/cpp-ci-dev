import numpy as np
import gymnasium
import time
from copy import deepcopy

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction
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
    Interaction that will repeat an action for some length of time, while the observation is still
    updated more frequently
    """
    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor
    ):
        super().__init__(cfg, env, state_constructor)
        self.steps_per_decision = cfg.steps_per_decision  # how many observation steps per decision step
        self.obs_length = cfg.obs_length  # how often to update the observation
        assert self.obs_length >= 0
        self.gamma = cfg.gamma

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[dict]]:
        state_list = [deepcopy(self.last_state)]
        observation_list = [deepcopy(self.last_obs)]
        reward_list = []
        terminated_list = []
        truncate_list = []
        env_info_list = []

        num_completed_steps = 0
        for obs_step in range(self.steps_per_decision):
            obs_end_time = time.time() + self.obs_length
            denormalized_action = self.action_normalizer.denormalize(action)
            next_observation, raw_reward, terminated, env_truncate, env_info = self.env.step(denormalized_action)

            truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
            reward = self.reward_normalizer(raw_reward)
            next_observation = self.obs_normalizer(next_observation)
            reward_list.append(reward)

            is_decision_point = (obs_step == self.steps_per_decision - 1)
            is_done = is_decision_point or terminated or truncate

            if is_done:
                next_state = self.state_constructor(next_observation, action, decision_point=True)
            else:
                next_state = self.state_constructor(next_observation, action)

            # update the lists
            state_list.append(next_state)
            observation_list.append(next_observation)
            terminated_list.append(terminated)
            truncate_list.append(truncate)
            env_info_list.append(env_info)

            # wait until enough time has elapsed
            if self.obs_length > 0:
                time.sleep(obs_end_time - time.time())

            if is_done:
                num_completed_steps = obs_step + 1  # plus one for zero-indexing of for loop
                break

        partial_return_list = [reward_list[-1]]
        for i in range(-2, -len(reward_list) - 1, -1):
            partial_return_list.insert(0, reward_list[i] + self.gamma * partial_return_list[i + 1])

        # assemble the transitions
        transitions = []
        for obs_step in range(num_completed_steps):  # note: we do not return a transition for the final state
            gamma_exp = num_completed_steps - obs_step
            transition = Transition(
                observation_list[obs_step],  # the current obs
                state_list[obs_step],  # the current state
                action,
                observation_list[obs_step + 1],  # the next observation
                state_list[obs_step + 1],  # the next state
                partial_return_list[obs_step],
                observation_list[-1],  # the boot observation
                state_list[-1],  # the boot state
                terminated_list[obs_step],
                terminated_list[obs_step],
                # state is a decision point at the state
                True if obs_step == 0 else False,
                # the next state is only a decision point at the end
                True if obs_step == num_completed_steps - 1 else False,
                gamma_exp
            )

            transitions.append(transition)

        self.last_state = next_state
        self.last_obs = next_observation

        return transitions, env_info_list
