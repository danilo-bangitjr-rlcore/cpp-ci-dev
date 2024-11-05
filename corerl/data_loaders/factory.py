from corerl.data_loaders.base import dl_group, old_dl_group, BaseDataLoaderConfig, BaseDataLoader, OldBaseDataLoader


def init_data_loader(cfg: BaseDataLoaderConfig) -> BaseDataLoader:
    return dl_group.dispatch(cfg)


def init_old_data_loader(cfg: BaseDataLoaderConfig) -> OldBaseDataLoader:
    return old_dl_group.dispatch(cfg)
