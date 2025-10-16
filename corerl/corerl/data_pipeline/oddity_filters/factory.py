from typing import Annotated

from pydantic import Field

from corerl.configs.data_pipeline.oddity_filters.conditional import ConditionalFilterConfig
from corerl.configs.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.configs.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.configs.data_pipeline.oddity_filters.stuck_detector import StuckDetectorConfig

OddityFilterConfig = Annotated[
    EMAFilterConfig
    | StuckDetectorConfig
    | ConditionalFilterConfig
    | IdentityFilterConfig
, Field(discriminator='name')]
