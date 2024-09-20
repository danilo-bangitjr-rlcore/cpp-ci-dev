from omegaconf import DictConfig
from corerl.data_loaders.base import BaseDataLoader
from corerl.data_loaders.direct_action import DirectActionDataLoader
from corerl.data_loaders.direct_action import OldDirectActionDataLoader

def init_data_loader(cfg: DictConfig) -> BaseDataLoader:
    """
    config files: corerl/config/data_loader
    """
    name = cfg.name

    if name == "direct_action":
        return OldDirectActionDataLoader(cfg)
    else:
        raise NotImplementedError

def init_data_loader_test(cfg: DictConfig) -> BaseDataLoader:
    """
    config files: corerl/config/data_loader
    """
    name = cfg.name

    if name == "direct_action":
        return OldDirectActionDataLoader(cfg), DirectActionDataLoader(cfg)
    else:
        raise NotImplementedError
