from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, cfg, state_dim, action_dim, discrete_control):
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.update_counter = 0 # TODO not sure if we should keep this here
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.discrete_control = discrete_control

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def update_buffer(self, transition):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError


class BaseAC(BaseAgent):
    @abstractmethod
    def update_actor(self):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self):
        raise NotImplementedError