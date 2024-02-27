import numpy as np


class Buffer:
    def __init__(self, memory, batch_size, seed=0):
        self.rng = np.random.RandomState(seed)
        self.memory = memory
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        # resets to start of buffer
        self.pos = (self.pos + 1) % self.memory

    def sample(self, batch_size=None):
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        # for i in range(len(batch_data)):
        #     if batch_data[i].ndim == 1:
        #         batch_data[i] = batch_data[i].reshape(-1, 1)
        batch_data = self.prepare_data(batch_data)
        return batch_data

    def sample_batch(self):
        if len(self.data) == 0:
            return None
        sampled_data = list(self.data)
        if len(sampled_data) > 1:
            batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        else:
            batch_data = [np.asarray([x]) for x in sampled_data[0]]
        batch_data = self.prepare_data(batch_data)
        return batch_data

    def prepare_data(self, batch_data):
        for i in range(len(batch_data)):
            if batch_data[i].ndim == 1:
                batch_data[i] = batch_data[i].reshape(-1, 1)
        return batch_data

    def load(self, states, actions, cumulants, dones, truncates):
        for i in range(len(states) - 1):
            self.feed([states[i], actions[i], cumulants[i], states[i+1], int(dones[i]), int(truncates[i])])

    @property
    def size(self):
        return len(self.data)

    def reset(self):
        self.data = []
        self.pos = 0

    def get_all_data(self):
        return self.data

    def update_priorities(self, priority=None):
        pass


class PriorityBuffer(Buffer):
    def __init__(self, memory, batch_size, seed=0):
        super(PriorityBuffer, self).__init__(memory, batch_size, seed)
        self.priority = []

    def feed(self, experience):
        super(PriorityBuffer, self).feed(experience)
        self.priority = list(self.priority)
        if self.pos >= len(self.data):
            self.priority.append(1.0)
        else:
            self.priority[self.pos] = 1.0
        self.priority = np.asarray(self.priority)
        self.priority /= self.priority.sum()

    def sample(self, batch_size=None):
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = self.rng.choice(self.size, batch_size, replace=False, p=self.priority)
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        batch_data = self.prepare_data(batch_data)
        return batch_data

    def update_priorities(self, priority=None):
        if priority is None:
            raise NotImplementedError
        else:
            self.priority = list(priority)

def init_buffer(name, cfg):
    if name == 'Prioritized':
        return PriorityBuffer(cfg.buffer_size, cfg.batch_size, cfg.seed)
    else:
        return Buffer(cfg.buffer_size, cfg.batch_size, cfg.seed)