import random
import numpy as np
from tqdm import tqdm
from collections import deque

from corerl.interaction.normalizer import NormalizerInteraction
from corerl.data import ObsTransition
from corerl.data import Transition, Trajectory
from corerl.alerts.composite_alert import CompositeAlert
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
    obs_transition.prev_action = interaction.action_normalizer(obs_transition.prev_action)
    obs_transition.obs = interaction.obs_normalizer(obs_transition.obs)
    obs_transition.action = interaction.action_normalizer(obs_transition.action)
    obs_transition.next_obs = interaction.obs_normalizer(obs_transition.next_obs)
    obs_transition.reward = interaction.reward_normalizer(obs_transition.reward)

    return obs_transition

def get_new_anytime_transitions(curr_decision_transitions, states, interaction, alerts):
    """
    Produce the agent and alert state transitions using the observation transitions that occur between two decision points
    """
    # Alerts can use different discount factors than the agent's value functions
    alert_gammas = np.array(alerts.get_discount_factors())

    rewards = [curr_obs_transition.reward for curr_obs_transition in curr_decision_transitions]
    next_obs_list = [curr_obs_transition.next_obs for curr_obs_transition in curr_decision_transitions]

    cumulants = []
    for i in range(len(rewards)):
        curr_cumulants = interaction.get_cumulants(rewards[i], next_obs_list[i])
        cumulants.append(curr_cumulants)

    new_agent_transitions = []
    new_alert_transitions = []

    # n_step = 0: bootstrap off state at next decision point
    # n_step > 0: bootstrap off state n steps into the future without crossing decision boundary
    if interaction.n_step == 0 or interaction.n_step >= interaction.steps_per_decision:
        n_step_rewards = deque([], interaction.steps_per_decision)
        n_step_cumulants = deque([], interaction.steps_per_decision)
        boot_state_queue = deque([], interaction.steps_per_decision)
        boot_obs_queue = deque([], interaction.steps_per_decision)
    else:
        n_step_rewards = deque([], interaction.n_step)
        n_step_cumulants = deque([], interaction.n_step)
        boot_state_queue = deque([], interaction.n_step)
        boot_obs_queue = deque([], interaction.n_step)

    boot_state_queue.appendleft(states[-1])
    boot_obs_queue.appendleft(curr_decision_transitions[-1].next_obs)

    dp_counter = 1
    # Iteratively create the agent and alert transitions
    for i in range(len(curr_decision_transitions) - 1, -1, -1):
        obs = curr_decision_transitions[i].obs
        state = states[i]
        action = curr_decision_transitions[i].action
        reward = curr_decision_transitions[i].reward
        s_dp = curr_decision_transitions[i].obs_dp
        next_obs = curr_decision_transitions[i].next_obs
        next_state = states[i+1]
        cumulant = cumulants[i]
        term = curr_decision_transitions[i].terminated
        trunc = curr_decision_transitions[i].truncate

        # Create Agent Transition
        np_n_step_rewards = interaction.update_n_step_cumulants(n_step_rewards, np.array([reward]), interaction.gamma)

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
            term,
            trunc,
            s_dp,
            ns_dp,
            gamma_exp)

        new_agent_transitions.append(agent_transition)

        # Create Alert Transition(s)
        np_n_step_cumulants = interaction.update_n_step_cumulants(n_step_cumulants, cumulant, alert_gammas)

        # Create transition for each alert type, using the relevant part of the cumulant
        step_alert_transitions = []
        alert_start_ind = 0
        for alert in alerts.alerts:
            alert_end_ind = alert_start_ind + alert.get_dim()

            alert_transition = Transition(
                obs,
                state,
                action,
                next_obs,  # the immediate next obs
                next_state,  # the immediate next state
                np_n_step_cumulants[-1][alert_start_ind : alert_end_ind].item(),
                boot_obs_queue[-1],  # the obs we bootstrap off
                boot_state_queue[-1],  # the state we bootstrap off
                term,
                trunc,
                s_dp,
                ns_dp,
                gamma_exp)

            step_alert_transitions.append(alert_transition)
            alert_start_ind = alert_end_ind

        new_alert_transitions.append(step_alert_transitions)

        # Update queues and counters
        dp_counter += 1
        boot_state_queue.appendleft(state)
        boot_obs_queue.appendleft(obs)
        n_step_rewards = deque(np_n_step_rewards, n_step_rewards.maxlen)
        n_step_cumulants = deque(np_n_step_cumulants, n_step_cumulants.maxlen)

    new_agent_transitions.reverse()
    new_alert_transitions.reverse()

    return new_agent_transitions, new_alert_transitions


def make_anytime_transitions_for_chunk(curr_chunk_obs_transitions, interaction, alerts,
                                       steps_per_decision, return_scs, gamma, sc_warmup):
    """
    Produce Anytime transitions for a continuous chunk of observation transitions (no data gaps)
    """
    curr_chunk_agent_transitions = []
    curr_chunk_alert_transitions = []
    new_scs = []

    sc = interaction.state_constructor
    sc.reset()

    # Using ObsTransition.next_obs to create remaining states so creating first state with ObsTransition.obs
    first_obs_transition = deepcopy(curr_chunk_obs_transitions[0])
    first_obs_transition = _normalize(first_obs_transition, interaction)
    state = sc(first_obs_transition.obs,
               first_obs_transition.prev_action,
               initial_state=True,
               decision_point=first_obs_transition.obs_dp,
               steps_since_decision=first_obs_transition.obs_steps_since_decision)
    states = [state]

    # Produce remaining states and create list of transitions when decision points are encountered
    curr_decision_obs_transitions = []
    for obs_transition in curr_chunk_obs_transitions:
        obs_transition = _normalize(obs_transition, interaction)

        next_state = sc(obs_transition.next_obs,
                        obs_transition.action,
                        initial_state=False,
                        decision_point=obs_transition.next_obs_dp,
                        steps_since_decision=obs_transition.next_obs_steps_since_decision)
        states.append(next_state)
        curr_decision_obs_transitions.append(obs_transition)

        if return_scs:
            new_scs.append(deepcopy(sc))

        # If at a decision point, create list of transitions for the states observed since the last decision point
        # if steps_per_decision is 1, curr_decision_obs_transitions could be empty
        if obs_transition.next_obs_dp and len(curr_decision_obs_transitions):
            assert len(states) == len(curr_decision_obs_transitions) + 1

            new_agent_transitions, new_alert_transitions = get_new_anytime_transitions(curr_decision_obs_transitions,
                                                                                       states, interaction, alerts)

            curr_chunk_agent_transitions += new_agent_transitions
            curr_chunk_alert_transitions += new_alert_transitions

            curr_decision_obs_transitions = []
            states = [next_state]

        state = next_state

    # Remove the transitions that were created during the state constructor warmup period
    curr_chunk_agent_transitions = curr_chunk_agent_transitions[sc_warmup:]
    curr_chunk_alert_transitions = curr_chunk_alert_transitions[sc_warmup:]

    assert len(curr_chunk_obs_transitions) == len(curr_chunk_agent_transitions) + sc_warmup

    if return_scs:
        new_scs = new_scs[sc_warmup:]
        assert len(new_scs) == len(curr_chunk_agent_transitions)

    return curr_chunk_agent_transitions, curr_chunk_alert_transitions, new_scs


def make_anytime_transitions(
        obs_transitions: list[ObsTransition],
        interaction: NormalizerInteraction,
        alerts: CompositeAlert,
        sc_warmup=0,
        steps_per_decision=30,
        gamma=0.9,
        return_scs=False):

    obs_transitions = deepcopy(obs_transitions)
    agent_transitions = []
    alert_transitions = []
    done = False
    transition_idx = 0
    pbar = tqdm(total=len(obs_transitions))
    scs = []
    while not done:
        # first, get transitions until a data gap
        curr_chunk_obs_transitions = []
        gap = False
        while not (gap or done):
            obs_transition = obs_transitions[transition_idx]
            curr_chunk_obs_transitions.append(obs_transition)
            gap = obs_transition.gap
            transition_idx += 1
            done = transition_idx == len(obs_transitions)
            pbar.update(1)

        new_agent_transitions, new_alert_transitions, new_scs = make_anytime_transitions_for_chunk(curr_chunk_obs_transitions, interaction, alerts,
                                                                                                   steps_per_decision, return_scs, gamma, sc_warmup)

        agent_transitions += new_agent_transitions
        alert_transitions += new_alert_transitions
        scs += new_scs

    return agent_transitions, alert_transitions, scs


def make_anytime_trajectories(
        obs_transitions: list[ObsTransition],
        interaction: NormalizerInteraction,
        alerts: CompositeAlert,
        sc_warmup=0,
        steps_per_decision=30,
        gamma=0.9,
        return_scs=False):

    obs_transitions = deepcopy(obs_transitions)
    trajectories = []
    alert_transitions = []
    done = False
    transition_idx = 0
    pbar = tqdm(total=len(obs_transitions))
    scs = []
    while not done:
        # first, get transitions until a data gap
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
        new_agent_transitions, new_alert_transitions, new_scs = make_anytime_transitions_for_chunk(curr_chunk_obs_transitions, interaction, alerts,
                                                                                                   steps_per_decision, return_scs, gamma, sc_warmup)
        alert_transitions += new_alert_transitions

        new_traj = Trajectory()
        for transition in new_agent_transitions:
            new_traj.add_transition(transition)
            new_traj.add_start_sc(new_scs[0])
        trajectories.append(new_traj)

    return trajectories, alert_transitions, scs
