from corerl.configs.data_pipeline.imputers.per_tag.base import BasePerTagImputerConfig
from corerl.data_pipeline.imputers.per_tag.base import BasePerTagImputer, per_tag_imputer_group


def init_per_tag_imputer(cfg: BasePerTagImputerConfig) -> BasePerTagImputer:
    return per_tag_imputer_group.dispatch(cfg)
