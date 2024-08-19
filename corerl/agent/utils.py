import torch
import numpy as np
from jaxtyping import Float
from typing import Optional

import corerl.component.network.utils as utils
from corerl.component.network.utils import to_np, tensor
from corerl.utils.device import device


def get_batch_actions_discrete(state_batch: Float[torch.Tensor, "batch_size state_dim"], action_dim: int,
                               samples=Optional[int]) \
        -> Float[torch.Tensor, "batch_size*action_dim action_dim"]:
    batch_size = state_batch.shape[0]
    if samples is None:
        actions = torch.arange(action_dim).reshape((1, -1))
        actions = actions.repeat_interleave(batch_size, dim=0).reshape((-1, 1))
    else:
        actions = torch.randint(high=action_dim, size=(batch_size * samples)).reshape((-1, 1))

    a_onehot = torch.FloatTensor(actions.size()[0], action_dim)
    a_onehot.zero_()
    sample_actions = a_onehot.scatter_(1, actions, 1)
    return sample_actions.to(device.device)


def get_top_action(func, states, actions, action_dim, batch_size, n_actions, return_idx=None):

    if return_idx is None:
        x = func([states], [actions])
    else:  # in func returns a tuple
        x = func([states], [actions])[0]

    x = x.reshape((batch_size, n_actions, 1))
    sorted_q_inds = torch.argsort(x, dim=1, descending=True)
    best_inds = sorted_q_inds[:, 0, :]  # index for the best action in each state
    best_inds = best_inds.unsqueeze(1)
    best_inds = best_inds.repeat_interleave(action_dim, -1)

    actions = actions.reshape(batch_size, n_actions, action_dim)
    best_actions = torch.gather(actions, dim=1, index=best_inds)
    return best_actions.to(device.device)


# def get_max_actions_critic(critic, states, actions, action_dim, batch_size, n_samples):
#     qs = critic.get_q(states, actions)
#     qs = qs.reshape((batch_size, n_samples, 1))
#     sorted_q_inds = torch.argsort(qs, dim=1, descending=True)
#
#     best_inds = sorted_q_inds[:, 0, :]  # index for the best action in each state
#     best_inds = best_inds.unsqueeze(1)
#     best_inds = best_inds.repeat_interleave(action_dim, -1)
#
#     actions = actions.reshape(batch_size, n_samples, action_dim)
#     best_actions = torch.gather(actions, dim=1, index=best_inds)
#     return best_actions

# def get_max_actions_actor(actor, states, actions, action_dim, batch_size, n_samples):
#     log_probs, _ = actor.get_log_prob(states, actions)
#     log_probs = log_probs.reshape((batch_size, n_samples, 1))
#     sorted_prob_inds = torch.argsort(log_probs, dim=1, descending=True)
#
#     best_inds = sorted_prob_inds[:, 0, :]  # index for the best action in each state
#     best_inds = best_inds.unsqueeze(1)
#     best_inds = best_inds.repeat_interleave(action_dim, -1)
#
#     sample_actions = actions.reshape(batch_size, n_samples, action_dim)
#     best_actions = torch.gather(sample_actions, dim=1, index=best_inds)
#     return best_actions


def get_test_state_qs_and_policy_params(agent, test_transitions):
    test_actions = 100
    num_states = len(test_transitions)
    test_states = []
    for transition in test_transitions:
        test_states.append(transition.state)
    test_states_np = np.array(test_states, dtype=np.float32)

    test_states = tensor(test_states_np, device)
    actions = np.linspace(np.array([0]), np.array([1]), num=test_actions)

    # Q-Values
    repeated_test_states = test_states.repeat_interleave(test_actions, dim=0)
    repeated_actions = [actions for i in range(num_states)]
    repeated_actions = np.concatenate(repeated_actions)
    repeated_actions = tensor(repeated_actions, device)

    q_values, ensemble_qs = agent.q_critic.get_qs([repeated_test_states], [repeated_actions], with_grad=False)
    q_values = to_np(q_values)
    q_values = q_values.reshape(num_states, test_actions)
    ensemble_qs = to_np(ensemble_qs)
    ensemble_qs = ensemble_qs.reshape(agent.q_critic.model.ensemble, num_states, test_actions)

    # Actor Params
    actor_alphas, actor_betas = agent.actor.model.get_dist_params(test_states)
    actor_alphas = to_np(actor_alphas)
    actor_betas = to_np(actor_betas)

    return test_states_np, actions, q_values, ensemble_qs, np.array(list(zip(actor_alphas, actor_betas)))

