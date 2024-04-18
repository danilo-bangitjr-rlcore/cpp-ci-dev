import torch
from jaxtyping import Float


def get_batch_actions_discrete(state_batch: Float[torch.Tensor, "batch_size state_dim"], action_dim: int) \
        -> Float[torch.Tensor, "batch_size*action_dim action_dim"]:
    batch_size = state_batch.shape[0]
    actions = torch.arange(action_dim).reshape((1, -1))
    actions = actions.repeat_interleave(batch_size, dim=0).reshape((-1, 1))
    a_onehot = torch.FloatTensor(actions.size()[0], action_dim)
    a_onehot.zero_()
    sample_actions = a_onehot.scatter_(1, actions, 1)
    return sample_actions
