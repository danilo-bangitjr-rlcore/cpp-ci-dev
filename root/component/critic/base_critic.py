from abc import ABC, abstractmethod

class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def update(self, loss):
        raise NotImplementedError


class BaseV(BaseCritic):
    @abstractmethod
    def __init__(self, cfg, state_dim):
        super(BaseV, self).__init__(cfg)

    @abstractmethod
    def get_v(self, state):
        raise NotImplementedError


class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg, state_dim, action_dim):
        super(BaseQ, self).__init__(cfg)

    @abstractmethod
    def get_q(self, state, action):
        raise NotImplementedError
