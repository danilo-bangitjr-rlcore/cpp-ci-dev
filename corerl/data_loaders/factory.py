from corerl.data_loaders.base import dl_group, BaseDataLoaderConfig, BaseDataLoader


def init_data_loader(cfg: BaseDataLoaderConfig) -> BaseDataLoader:
    return dl_group.dispatch(cfg)