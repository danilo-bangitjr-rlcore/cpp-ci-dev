from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
import corerl.data_pipeline.imputers.exp_moving_detector  # noqa: F401
import corerl.data_pipeline.imputers.identity  # noqa: F401


def init_imputer(cfg: BaseImputerConfig) -> BaseImputer:
    return imputer_group.dispatch(cfg)
