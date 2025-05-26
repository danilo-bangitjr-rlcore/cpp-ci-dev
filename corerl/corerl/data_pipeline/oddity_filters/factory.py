from pydantic import Field
from typing_extensions import Annotated

from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.oddity_filters.stuck_detector import StuckDetectorConfig
from corerl.state import AppState

OddityFilterConfig = Annotated[
    EMAFilterConfig
    | StuckDetectorConfig
    | IdentityFilterConfig
, Field(discriminator='name')]

def init_oddity_filter(cfg: BaseOddityFilterConfig, app_state: AppState) -> BaseOddityFilter:
    return outlier_group.dispatch(cfg, app_state)
