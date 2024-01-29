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

    def sample(self):
        if len(self.data) == 0:
            return None
        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(self.batch_size)]

        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        for i in range(len(batch_data)):
            if batch_data[i].ndim == 1:
                batch_data[i] = batch_data[i].reshape(-1, 1)
        return batch_data

    @property
    def size(self):
        return len(self.data)

    def reset(self):
        self.data = []
        self.pos = 0

    def get_all_data(self):
        return self.data