import random
from tqdm import tqdm


from corerl.interaction.normalizer import NormalizerInteraction
from corerl.data import Transition, Trajectory, ObsTransition
from copy import deepcopy


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


def _normalize(obs_transition, interaction):
    obs_transition.obs = interaction.obs_normalizer(obs_transition.obs)
    obs_transition.action = interaction.action_normalizer(obs_transition.action)
    obs_transition.reward = interaction.reward_normalizer(obs_transition.reward)
    obs_transition.next_obs = interaction.obs_normalizer(obs_transition.next_obs)
    return obs_transition


def get_new_anytime_transitions(curr_decision_transitions, gamma, states, obs_transition, last_state):
    rewards = [curr_obs_transition.reward for curr_obs_transition in curr_decision_transitions]
    partial_returns = [rewards[-1]]
    for i in range(-2, -len(curr_decision_transitions) - 1, -1):
        partial_returns.insert(0, rewards[i] + partial_returns[-1] * gamma)

    max_gamma_exp = len(curr_decision_transitions)
    new_transitions = []
    last_obs = curr_decision_transitions[-1].next_obs
    for curr_obs_i, curr_obs_transition in enumerate(curr_decision_transitions):
        transition = Transition(
            curr_obs_transition.obs,
            states[curr_obs_i],
            curr_obs_transition.action,
            curr_obs_transition.next_obs,
            states[curr_obs_i + 1],
            partial_returns[curr_obs_i],
            # the obs for bootstrapping comes from obs_transition, since it is a decision point
            last_obs,  # obs_transition.obs,
            last_state,  # the state for bootstrapping
            False,  # assume continuing
            False,  # assume continuing
            curr_obs_i == 0,  # only the first state is a decision point
            True,
            max_gamma_exp - curr_obs_i)  # TODO: the gamme exponent is wrong here
        new_transitions.append(transition)

    return new_transitions


def make_anytime_transitions_for_chunk(curr_chunk_obs_transitions, interaction, sc,
                                       steps_per_decision, return_scs, gamma, sc_warmup):
    new_transitions = []
    new_scs = []
    steps_since_decision = 0
    first_obs_transition = deepcopy(curr_chunk_obs_transitions[0])
    first_obs_transition = _normalize(first_obs_transition, interaction)

    sc.reset()
    last_state = sc(first_obs_transition.obs,
                    first_obs_transition.action,
                    initial_state=True,
                    decision_point=True,
                    steps_since_decision=steps_since_decision)
    states = [last_state]
    last_action = first_obs_transition.action
    curr_chunk_transitions = []
    curr_decision_obs_transitions = []

    if return_scs:
        new_scs.append(deepcopy(sc))

    # next, consider remaining transitions and compute the next states
    for obs_transition in curr_chunk_obs_transitions:
        obs_transition = _normalize(obs_transition, interaction)
        decision_point = (obs_transition.action != last_action
                          or steps_since_decision >= steps_per_decision - 1)

        next_state = sc(obs_transition.next_obs,
                        obs_transition.action,
                        initial_state=False,
                        decision_point=decision_point,
                        steps_since_decision=steps_since_decision)

        states.append(next_state)
        if return_scs:
            new_scs.append(deepcopy(sc))

        # if steps_per_decision is 1, curr_decision_obs_transitions could be empty
        if decision_point and len(curr_decision_obs_transitions):
            # because we have already produced the next state for the observation after the decision point,
            # there are two more states than curr_decision_obs_transitions
            assert len(states) == len(curr_decision_obs_transitions) + 2

            new_transitions = get_new_anytime_transitions(curr_decision_obs_transitions, gamma,
                                                          states, obs_transition, last_state)

            curr_chunk_transitions += new_transitions
            # if obs_transitions was a decision point, we did not add it previously
            curr_decision_obs_transitions = [obs_transition]
            states = [last_state, next_state]
            last_action = obs_transition.action
            steps_since_decision = 0
        else:
            # if obs_transitions was not a decision point we will add it now
            curr_decision_obs_transitions.append(obs_transition)
            steps_since_decision += 1

        last_state = next_state

    # add remaining transitions for the final iteration
    new_transitions = get_new_anytime_transitions(curr_decision_obs_transitions, gamma,
                                                  states, obs_transition, last_state)
    curr_chunk_transitions += new_transitions
    curr_chunk_transitions = curr_chunk_transitions[sc_warmup:]  # only add transitions after the warmup period

    assert len(curr_chunk_obs_transitions) == len(curr_chunk_transitions) + sc_warmup

    if return_scs:
        # exclude the last element, since it corresponds to a state after the last transition
        new_scs = new_scs[sc_warmup:-1]
        assert len(new_scs) == len(curr_chunk_transitions)

    return curr_chunk_transitions, new_scs


def make_anytime_transitions(
        obs_transitions: list[ObsTransition],
        interaction: NormalizerInteraction,
        sc_warmup=0,
        steps_per_decision=30,
        gamma=0.9,
        return_scs=False
):
    obs_transitions = deepcopy(obs_transitions)
    sc = interaction.state_constructor
    transitions = []
    done = False
    transition_idx = 0
    pbar = tqdm(total=len(obs_transitions))
    scs = []
    while not done:
        # first, get transitions until a gap
        curr_chunk_obs_transitions = []
        gap = False
        while not (gap or done):
            obs_transition = obs_transitions[transition_idx]
            curr_chunk_obs_transitions.append(obs_transition)
            gap = obs_transition.gap
            transition_idx += 1
            done = transition_idx == len(obs_transitions)
            pbar.update(1)

        new_transitions, new_scs = make_anytime_transitions_for_chunk(curr_chunk_obs_transitions, interaction, sc,
                                                                      steps_per_decision, return_scs, gamma, sc_warmup)
        transitions += new_transitions
        scs += new_scs

    return transitions, scs


def make_anytime_trajectories(
        obs_transitions: list[ObsTransition],
        interaction: NormalizerInteraction,
        sc_warmup=0,
        steps_per_decision=30,
        gamma=0.9,
        return_scs=False
):
    obs_transitions = deepcopy(obs_transitions)
    sc = interaction.state_constructor
    trajectories = []
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

        # we will store all
        new_transitions, new_scs = make_anytime_transitions_for_chunk(curr_chunk_obs_transitions, interaction, sc,
                                                                      steps_per_decision, return_scs, gamma, sc_warmup)

        new_traj = Trajectory()
        for i, transition in enumerate(new_transitions):
            new_traj.add_transition(transition)
            if return_scs:
                new_traj.add_start_sc(new_scs[0])
                new_traj.add_sc(new_scs[i])

        trajectories.append(new_traj)

    return trajectories
