from dataclasses import dataclass, field
from omegaconf import MISSING

from corerl.data_pipeline.imputers.base import BaseImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilterConfig
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig
from corerl.data_pipeline.state_constructors.components.norm import NormalizerConfig
from corerl.utils.hydra import list_


@dataclass
class TagConfig:
    name: str = MISSING

    bounds: tuple[float | None, float | None] = (None, None)
    outlier: BaseOddityFilterConfig = field(default_factory=EMAFilterConfig)
    imputer: BaseImputerConfig = field(default_factory=IdentityImputerConfig)
    state_constructor: list[BaseTransformConfig] = list_([NormalizerConfig()])
    is_action: bool = False
