from abc import ABC, abstractmethod

class BaseEval(ABC):
    @abstractmethod
    def __init__(self, cfg, **kwargs):
        raise NotImplementedError
    @abstractmethod
    def do_eval(self, cfg, **kwargs):
        raise NotImplementedError
    @abstractmethod
    def get_stats(self):
        raise NotImplementedError

