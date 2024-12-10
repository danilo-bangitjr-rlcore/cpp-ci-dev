from corerl.data_pipeline import dl_group
from corerl.data_pipeline.base import BaseDataLoaderConfig, BaseDataLoader, OldBaseDataLoader


def init_data_loader(cfg: BaseDataLoaderConfig) -> BaseDataLoader | OldBaseDataLoader:
    return dl_group.dispatch(cfg)
