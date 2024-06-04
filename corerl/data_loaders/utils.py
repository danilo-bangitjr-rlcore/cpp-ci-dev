import numpy as np
import random
from tqdm import tqdm
from dataclasses import dataclass, fields

from corerl.interaction.normalizer import NormalizerInteraction
from corerl.data import Transition
from copy import deepcopy


@dataclass
class ObsTransition:
    obs: np.array  # the raw observation of state
    action: np.array
    reward: float
    next_obs: np.array  # the immediate next observation
    terminated: bool
    truncate: bool
    gap: bool  # whether there is a gap in the dataset following next_ovs

    def __iter__(self):
        for field in fields(self):
            yield getattr(self, field.name)

    @property
    def field_names(self):
        return [field.name for field in fields(self)]


def train_test_split(*lsts, train_split: float = 0.9, shuffle: bool = True) -> (list[tuple], list[tuple]):
    num_samples = len(lsts[0])
    for a in lsts:
        assert len(a) == num_samples

    if shuffle:
        lsts = parallel_shuffle(*lsts)

    num_train_samples = int(train_split * num_samples)
    train_samples = [lsts[:num_train_samples] for lsts in lsts]
    test_samples = [lsts[num_train_samples:] for lsts in lsts]

    return list(zip(train_samples, test_samples))


def parallel_shuffle(*args):
    zipped_list = list(zip(*args))
    random.shuffle(zipped_list)
    unzipped = zip(*zipped_list)
    return list(unzipped)


def make_transitions(
        obs_transitions: list[ObsTransition],
        interaction: NormalizerInteraction,
        sc_warmup=0
):
    sc = interaction.state_constructor
    transitions = []
    initial_state = True
    last_state = None
    warmup_counter = 0
    for obs_transition in obs_transitions:
        obs = obs_transition.obs
        action = obs_transition.action
        reward = obs_transition.reward
        next_obs = obs_transition.next_obs

        norm_obs = interaction.obs_normalizer(obs)
        norm_next_obs = interaction.obs_normalizer(next_obs)
        norm_action = interaction.action_normalizer(action)
        norm_reward = interaction.reward_normalizer(reward)

        if initial_state:
            # Note: assumes the last action was the same as the current action. This won't always be the case
            last_state = sc(norm_obs, norm_action, initial_state=True)
            initial_state = False

        next_state = sc(norm_next_obs, norm_action, initial_state=False)

        warmup_counter += 1
        if warmup_counter >= sc_warmup:
            transition = Transition(
                obs,
                last_state,
                norm_action,
                norm_next_obs,
                next_state,
                norm_reward,
                norm_next_obs,  # the obs for bootstrapping is the same as the next obs here
                next_state,  # the state for bootstrapping is the same as the next state here
                obs_transition.terminated,
                obs_transition.truncate,
                True,  # last_state always a decision point
                True,  # next_state always a decision point
                1)

            transitions.append(transition)

        if obs_transition.gap:
            initial_state = True
            warmup_counter = 0

        last_state = next_state


def _normalize(obs_transition, interaction):
    obs_transition.obs = interaction.obs_normalizer(obs_transition.obs)
    obs_transition.action = interaction.action_normalizer(obs_transition.action)
    obs_transition.reward = interaction.reward_normalizer(obs_transition.reward)
    obs_transition.next_obs = interaction.obs_normalizer(obs_transition.next_obs)
    return obs_transition


def get_new_transitions(curr_decision_transitions, gamma, states, obs_transition, last_state):
    rewards = [curr_obs_transition.reward for curr_obs_transition in curr_decision_transitions]
    partial_returns = [rewards[-1]]
    for i in range(-2, -len(curr_decision_transitions) - 1, -1):
        partial_returns.insert(0, rewards[i] + partial_returns[-1] * gamma)

    max_gamma_exp = len(curr_decision_transitions) - 1
    new_transitions = []
    for curr_obs_i, curr_obs_transition in enumerate(curr_decision_transitions):
        transition = Transition(
            curr_obs_transition.obs,
            states[curr_obs_i],
            curr_obs_transition.action,
            curr_obs_transition.next_obs,
            states[curr_obs_i + 1],
            partial_returns[curr_obs_i],
            obs_transition.obs,
            # the obs for bootstrapping comes from obs_transition, since it is a decision point
            last_state,  # the state for bootstrapping
            False,  # assume continuing
            False,  # assume continuing
            curr_obs_i == 0,  # only the first state is a decision point
            True,
            max_gamma_exp - curr_obs_i)
        new_transitions.append(transition)
    return new_transitions


def make_anytime_transitions(
        obs_transitions: list[ObsTransition],
        interaction: NormalizerInteraction,
        sc_warmup=0,
        steps_per_decision=30,
        gamma=0.9
):
    obs_transitions = deepcopy(obs_transitions)
    sc = interaction.state_constructor
    transitions = []
    done = False
    transition_idx = 0
    pbar = tqdm(total=len(obs_transitions))
    while not done:
        # first, get transitions until a gap
        curr_chunk_obs_transitions = []
        gap = False
        while not (gap or done):
            obs_transition = obs_transitions[transition_idx]
            curr_chunk_obs_transitions.append(obs_transition)
            gap = obs_transition.gap
            transition_idx += 1
            done = (transition_idx == len(obs_transitions) - 1)
            pbar.update(1)

        # At this point, we have all transitions until a gap happens
        # Initialize the state constructor with the first observation.
        steps_since_decision = 0
        first_obs_transition = deepcopy(curr_chunk_obs_transitions[0])
        first_obs_transition = _normalize(first_obs_transition, interaction)
        last_state = sc(first_obs_transition.obs,
                        first_obs_transition.action,
                        initial_state=True,
                        decision_point=True,
                        steps_since_decision=steps_since_decision)
        states = [last_state]
        last_action = first_obs_transition.action
        curr_chunk_transitions = []
        curr_decision_obs_transitions = []

        # next, consider remaining transitions and compute the next states
        for obs_transition in curr_chunk_obs_transitions:
            obs_transition = _normalize(obs_transition, interaction)
            decision_point = (obs_transition.action != last_action
                              or steps_since_decision >= steps_per_decision-1)

            next_state = sc(obs_transition.next_obs,
                            obs_transition.action,
                            initial_state=False,
                            decision_point=decision_point,
                            steps_since_decision=steps_since_decision)
            states.append(next_state)

            if decision_point:
                new_transitions = get_new_transitions(curr_decision_obs_transitions, gamma, states, obs_transition, last_state)
                curr_chunk_transitions += new_transitions
                curr_decision_obs_transitions = [obs_transition]   # if obs_transitions was a decision point, we did not add it previously
                states = [next_state]
                last_action = obs_transition.action
                steps_since_decision = 0
            else:
                curr_decision_obs_transitions.append(obs_transition)   # if obs_transitions was not a decision point we will add it now
                steps_since_decision += 1

            last_state = next_state
        transitions += curr_chunk_transitions[sc_warmup:] # only add transitions after the warmup period

    return transitions