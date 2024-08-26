import numpy
import torch
from corerl.component.network.utils import tensor, to_np
from corerl.utils.device import device
import gymnasium


class TorchWrapper:
    """
    Acts as an interface between gym environments which expect actions in the form of numpy arrays
    and agents who take actions in the form of torch tensors.
    """
    def __init__(self, env: gymnasium.Env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _process_state(self, state: numpy.ndarray) -> torch.Tensor:
        state = tensor(state.reshape((1, -1)), device.device)
        return state

    def reset(self) -> (torch.Tensor, dict):
        state, info = self.env.reset()
        state = self._process_state(state)
        return state, info

    def step(self, action: numpy.ndarray | torch.Tensor):
        action = to_np(action)[0] # TODO: not sure if i like the [0] here
        state, reward, done, truncate, env_info = self.env.step(action)
        state = self._process_state(state)
        return state, reward, done, truncate, env_info
