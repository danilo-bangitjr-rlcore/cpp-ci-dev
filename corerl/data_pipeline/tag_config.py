from dataclasses import dataclass, field
from omegaconf import MISSING

from corerl.data_pipeline.imputers.base import BaseImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetectorConfig
from corerl.data_pipeline.outlier_detectors.exp_moving_detector import ExpMovingDetectorConfig
from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig
from corerl.utils.hydra import list_


@dataclass
class TagConfig:
    name: str = MISSING

    bounds: tuple[float | None, float | None] = (None, None)
    outlier: BaseOutlierDetectorConfig = field(default_factory=ExpMovingDetectorConfig)
    imputer: BaseImputerConfig = field(default_factory=IdentityImputerConfig)
    state_constructor: list[BaseTransformConfig] = list_()
