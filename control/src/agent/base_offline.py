import numpy as np
from src.agent.base import BaseAC
import src.network.torch_utils as torch_utils


class BaseACOff(BaseAC):
    def __init__(self, cfg):
        super(BaseACOff, self).__init__(cfg)
        self.dataset, self.dataset_size = self.load_offline_data()

    def load_offline_data(self):
        data = self.env.get_dataset()
        """
        1. Normalization
        2. Tensor
        """
        dataset = {
            'obs': torch_utils.tensor(self.state_constructor(data['observations']), self.device),
            'act': torch_utils.tensor(self.action_normalizer(data['actions']), self.device),
            'reward': torch_utils.tensor(self.reward_normalizer(data['rewards']), self.device),
            'obs2': torch_utils.tensor(self.state_constructor(data['next_observations']), self.device),
            'done': torch_utils.tensor(data['terminals'], self.device)
        }
        return dataset, len(data['observations'])

    def step(self):
        self.update()
        self.update_stats(0, False, False)
        return

    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        idx = self.rng.choice(self.dataset_size, batch_size, replace=False)
        data = {
            'obs': self.dataset['obs'][idx],
            'act': self.dataset['act'][idx],
            'reward': self.dataset['reward'][idx],
            'obs2': self.dataset['obs2'][idx],
            'done': self.dataset['done'][idx],
            'trunc': self.dataset['done'][idx],
        }
        return data

    def get_all_data(self):
        return self.dataset
