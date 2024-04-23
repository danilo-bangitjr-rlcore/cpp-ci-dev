import torch
from jaxtyping import Float
from typing import Optional

def get_batch_actions_discrete(state_batch: Float[torch.Tensor, "batch_size state_dim"], action_dim: int, samples=Optional[int]) \
        -> Float[torch.Tensor, "batch_size*action_dim action_dim"]:
    batch_size = state_batch.shape[0]
    if samples is None:
        actions = torch.arange(action_dim).reshape((1, -1))
        actions = actions.repeat_interleave(batch_size, dim=0).reshape((-1, 1))
    else:
        actions = torch.randint(high=action_dim, size=(batch_size*samples)).reshape((-1, 1))

    a_onehot = torch.FloatTensor(actions.size()[0], action_dim)
    a_onehot.zero_()
    sample_actions = a_onehot.scatter_(1, actions, 1)
    return sample_actions


