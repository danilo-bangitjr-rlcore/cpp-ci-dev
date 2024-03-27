from abc import ABC, abstractmethod
from root.component.optimizers.factory import  init_optimizer
from root.component.networks.factory import  init_network
class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg):
        self.model = init_network(cfg)
        self.target = None
        self.optimizer = init_optimizer(cfg)

    @abstractmethod
    def update(self, loss):
        raise NotImplementedError


class BaseV(BaseCritic):
    @abstractmethod
    def __init__(self, cfg):
        super(BaseV, self).__init__(cfg)

    @abstractmethod
    def get_v(self, state):
        raise NotImplementedError

class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg):
        super(BaseQ, self).__init__(cfg)

    @abstractmethod
    def get_q(self, state, action):
        raise NotImplementedError
