from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, cfg):
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.update_counter = 0

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError

    def update(self):
        if self.update_counter % self.update_freq == 0:
            for _ in range(self.replay_ratio):
                self.atomic_update()
        self.update_counter += 1

    @abstractmethod
    def atomic_update(self):
        raise NotImplementedError

    # @abstractmethod
    # def update_actor(self):
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def update_critic(self):
    #     raise NotImplementedError

    @abstractmethod
    def update_buffer(self, transition):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError
