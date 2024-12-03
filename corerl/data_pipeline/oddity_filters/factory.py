from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
import corerl.data_pipeline.oddity_filters.ema_filter  # noqa: F401
import corerl.data_pipeline.oddity_filters.identity  # noqa: F401


def init_oddity_filter(cfg: BaseOddityFilterConfig) -> BaseOddityFilter:
    return outlier_group.dispatch(cfg)
