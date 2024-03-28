from abc import ABC, abstractmethod


class BaseActor(ABC):
    @abstractmethod
    def __init__(self, cfg):
        raise NotImplementedError

    # TODO may not keep get_action. Just including it for now in case we
    # want to implement stuff like epsilon-greedy
    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError
