import torch
from jaxtyping import Float
from typing import Optional


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
    return sample_actions


def get_top_action(func, states, actions, action_dim, batch_size, n_actions, return_idx=None):

    if return_idx is None:
        x = func(states, actions)
    else:  # in func returns a tuple
        x = func(states, actions)[0]

    x = x.reshape((batch_size, n_actions, 1))
    sorted_q_inds = torch.argsort(x, dim=1, descending=True)
    best_inds = sorted_q_inds[:, 0, :]  # index for the best action in each state
    best_inds = best_inds.unsqueeze(1)
    best_inds = best_inds.repeat_interleave(action_dim, -1)

    actions = actions.reshape(batch_size, n_actions, action_dim)
    best_actions = torch.gather(actions, dim=1, index=best_inds)
    return best_actions



