from typing import Callable
from dataclasses import dataclass, field
from pandas import DataFrame

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker

from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector, BaseOutlierDetectorConfig
from corerl.data_pipeline.outlier_detectors.identity import IdentityDetectorConfig
from corerl.data_pipeline.outlier_detectors.factory import init_outlier_detector

from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig
from corerl.data_pipeline.imputers.identity import IdentityImputerConfig
from corerl.data_pipeline.imputers.factory import init_imputer

from corerl.data_pipeline.transition_creators.dummy import DummyTransitionCreatorConfig
from corerl.data_pipeline.transition_creators.base import BaseTransitionCreator
from corerl.data_pipeline.transition_creators.factory import init_transition_creator

from corerl.data_pipeline.state_constructors.base import BaseStateConstructor, BaseStateConstructorConfig
from corerl.data_pipeline.state_constructors.identity import IdentityStateConstructorConfig
from corerl.data_pipeline.state_constructors.factory import init_state_constructor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import Transition

from corerl.data_pipeline.pipeline_utils import warmup_pruning, handle_data_gaps

type missing_data_checker_type = Callable[[PipelineFrame, TagConfig], PipelineFrame]
type bound_checker_type = Callable[[PipelineFrame, TagConfig], PipelineFrame]
type warmup_pruning_type = Callable[[list[Transition], int], list[Transition]]

WARMUP = 0


@dataclass
class PipelineConfig:
    outlier_detector: BaseOutlierDetectorConfig = field(default_factory=IdentityDetectorConfig)
    imputer: BaseImputerConfig = field(default_factory=IdentityImputerConfig)
    transition_creator: DummyTransitionCreatorConfig = field(default_factory=DummyTransitionCreatorConfig)
    state_constructor: BaseStateConstructorConfig = field(default_factory=IdentityStateConstructorConfig)


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        """
        This will be defined according to the Pipeline config eventually
        """
        self.missing_data: missing_data_checker_type = missing_data_checker
        self.bound_checker: bound_checker_type = bound_checker
        self.outlier_detector: BaseOutlierDetector = init_outlier_detector(cfg.outlier_detector)
        self.imputer: BaseImputer = init_imputer(cfg.imputer)
        self.transition_creator: BaseTransitionCreator = init_transition_creator(cfg.transition_creator)
        self.state_constructor: BaseStateConstructor = init_state_constructor(cfg.state_constructor)
        self.warmup_pruning: warmup_pruning_type = warmup_pruning

    def __call__(self, data: DataFrame, cfg: TagConfig) -> list[Transition]:
        pf = PipelineFrame(data)
        pf = self.missing_data(pf, cfg)
        pf = self.bound_checker(pf, cfg) # Will need to be a list of TagConfigs
        pf = self.outlier_detector(pf, cfg)
        pf = self.imputer(pf, cfg)
        pfs = handle_data_gaps(pf)
        transitions = []
        for pf in pfs:
            pf_transitions = self.transition_creator(pf, cfg)
            pf_transitions = self.state_constructor(pf_transitions)
            pf_transitions = self.warmup_pruning(pf_transitions, WARMUP)  # placeholder for WARMUP. Comes from somewhere
            transitions += pf_transitions
        return transitions
