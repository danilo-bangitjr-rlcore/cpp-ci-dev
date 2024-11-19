import numpy as np

from corerl.data_pipeline.datatypes import ObsTransition

def make_simple_obs_sequence(num_observations: int) -> list[np.ndarray]:
    obs_sequence = [np.array([1])]
    for _ in range(num_observations - 1):
        obs_sequence.append(np.array([0]))
    return obs_sequence


def make_simple_obs_transition_sequence(num_observations: int):
    observations = make_simple_obs_sequence(num_observations)
    action = np.array([1])
    reward = 1

    obs_transitions: list[ObsTransition] = []
    for i in range(len(observations) - 1):
        new_obs_transition = ObsTransition(
            obs=observations[i],
            action=action,
            reward=reward,
            next_obs=observations[i + 1],
            terminated=False,
            truncate=False,
            gap=False
        )
        obs_transitions.append(new_obs_transition)

    return obs_transitions
