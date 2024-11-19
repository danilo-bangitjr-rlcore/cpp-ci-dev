from typing import Callable
from dataclasses import dataclass
from pandas import DataFrame

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.missing_data_checkers import identity_missing_data_checker
from corerl.data_pipeline.bound_checkers import identity_bound_checker
from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector
from corerl.data_pipeline.outlier_detectors.identity import IdentityDetector
from corerl.data_pipeline.imputers.base import BaseImputer
from corerl.data_pipeline.imputers.identity import IdentityImputer
from corerl.data_pipeline.transition_creators.base import BaseTransitionCreator
from corerl.data_pipeline.state_constructors.base import BaseStateConstructor
from corerl.data_pipeline.state_constructors.identity import IdentityStateConstructor


from corerl.data_pipeline.datatypes import Transition

type missing_data_checker_type = Callable[[PipelineFrame, TagConfig], PipelineFrame]
type bound_checker_type = Callable[[PipelineFrame, TagConfig], PipelineFrame]


@dataclass
class PipelineConfig:
    pass


@dataclass  # Placeholder
class TagConfig:
    pass


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        """
        This will be defined according to the Pipeline config eventually
        """
        self.missing_data: missing_data_checker_type = identity_missing_data_checker
        self.bound_checker: bound_checker_type = identity_bound_checker
        self.outlier_detector: BaseOutlierDetector = IdentityDetector()
        self.imputer: BaseImputer = IdentityImputer()
        self.transition_creator: BaseTransitionCreator = BaseTransitionCreator()
        self.state_constructor: BaseStateConstructor = IdentityStateConstructor()

    def __call__(self, data: DataFrame, cfg: TagConfig) -> list[Transition]:
        pf = PipelineFrame(data)
        pf = self.missing_data(pf, cfg)
        pf = self.bound_checker(pf, cfg)
        pf = self.outlier_detector(pf, cfg)
        pf = self.imputer(pf, cfg)
        transitions = self.transition_creator(pf, cfg)
        transitions_with_state = self.state_constructor(transitions)
        return transitions_with_state
