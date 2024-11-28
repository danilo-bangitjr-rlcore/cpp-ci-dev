from corerl.data_pipeline.base import dl_group, BaseDataLoaderConfig, BaseDataLoader, OldBaseDataLoader


def init_data_loader(cfg: BaseDataLoaderConfig) -> BaseDataLoader | OldBaseDataLoader:
    return dl_group.dispatch(cfg)
