from corerl.data_pipeline import dl_group, register
from corerl.data_pipeline.base import BaseDataLoader, BaseDataLoaderConfig, OldBaseDataLoader


def init_data_loader(cfg: BaseDataLoaderConfig) -> BaseDataLoader | OldBaseDataLoader:
    register()
    return dl_group.dispatch(cfg)
