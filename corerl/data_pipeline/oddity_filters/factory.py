from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig

OddityFilterConfig = (
    EMAFilterConfig
    | IdentityFilterConfig
)

def init_oddity_filter(cfg: BaseOddityFilterConfig) -> BaseOddityFilter:
    return outlier_group.dispatch(cfg)
