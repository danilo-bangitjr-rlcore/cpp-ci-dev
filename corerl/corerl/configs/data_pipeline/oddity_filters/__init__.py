"""Oddity filter configuration union type."""

from corerl.configs.data_pipeline.oddity_filters.base import BaseOddityFilterConfig
from corerl.configs.data_pipeline.oddity_filters.conditional import ConditionalFilterConfig
from corerl.configs.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.configs.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.configs.data_pipeline.oddity_filters.stuck_detector import StuckDetectorConfig

OddityFilterConfig = (
    IdentityFilterConfig
    | EMAFilterConfig
    | StuckDetectorConfig
    | ConditionalFilterConfig
    | BaseOddityFilterConfig
)
