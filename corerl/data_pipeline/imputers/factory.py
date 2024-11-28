from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
import corerl.data_pipeline.imputers.identity  # noqa: F401
import corerl.data_pipeline.imputers.copy # noqa: F401
from corerl.data_pipeline.tag_config import TagConfig


def init_imputer(cfg: BaseImputerConfig, tag_cfg: TagConfig) -> BaseImputer:
    return imputer_group.dispatch(cfg, tag_cfg)
