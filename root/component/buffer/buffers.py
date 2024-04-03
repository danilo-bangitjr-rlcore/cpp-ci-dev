import numpy as np
import root.component.network.utils as network_utils
from omegaconf import DictConfig

def send_to_device(batch: list, device: str) -> dict:
    states, actions, rewards, next_states, terminals, truncations = batch
    s = network_utils.tensor(states, device)
    a = network_utils.tensor(actions, device)
    r = network_utils.tensor(rewards, device)
    ns = network_utils.tensor(next_states, device)
    d = network_utils.tensor(terminals, device)
    t = network_utils.tensor(truncations, device)
    data = {
        'states': s,
        'actions': a,
        'rewards': r,
        'next_states': ns,
        'dones': d,
        'truncs': t,
    }
    return data
class UniformBuffer:
    def __init__(self, cfg: DictConfig):
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.data = []
        self.pos = 0

    def feed(self, experience: tuple) -> None:
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        # resets to start of buffer
        self.pos = (self.pos + 1) % self.memory

    def sample(self, batch_size: int=None)-> dict:
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        batch_data = self.prepare_data(batch_data)
        batch_data = send_to_device(batch_data, device=self.device)
        return batch_data


    def sample_batch(self) -> dict:
        if len(self.data) == 0:
            return None
        sampled_data = list(self.data)
        if len(sampled_data) > 1:
            batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        else:
            batch_data = [np.asarray([x]) for x in sampled_data[0]]
        batch_data = self.prepare_data(batch_data)
        batch_data = send_to_device(batch_data, device=self.device)
        return batch_data

    def prepare_data(self, batch_data: list) -> list:
        for i in range(len(batch_data)):
            if batch_data[i].ndim == 1:
                batch_data[i] = batch_data[i].reshape(-1, 1)
        return batch_data

    def load(self, states: list, actions: list, cumulants: list, dones: list, truncates: list) -> None:
        for i in range(len(states) - 1):
            self.feed((states[i], actions[i], cumulants[i], states[i+1], int(dones[i]), int(truncates[i])))

    @property
    def size(self) -> int:
        return len(self.data)

    def reset(self) -> None:
        self.data = []
        self.pos = 0

    def get_all_data(self) -> list:
        return self.data

    def update_priorities(self, priority=None):
        pass


class PriorityBuffer(UniformBuffer):
    def __init__(self, cfg: DictConfig):
        super(PriorityBuffer, self).__init__(cfg)
        self.priority = []

    def feed(self, experience: tuple) -> None:
        super(PriorityBuffer, self).feed(experience)
        self.priority = list(self.priority)
        if self.pos >= len(self.data):
            self.priority.append(1.0)
        else:
            self.priority[self.pos] = 1.0
        self.priority = np.asarray(self.priority)
        self.priority /= self.priority.sum()

    def sample(self, batch_size: int=None) -> dict:
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = self.rng.choice(self.size, batch_size, replace=False, p=self.priority)
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        batch_data = self.prepare_data(batch_data)
        batch_data = send_to_device(batch_data, device=self.device)
        return batch_data

    def update_priorities(self, priority=None):
        if priority is None:
            raise NotImplementedError
        else:
            self.priority = list(priority)
