from collections.abc import Mapping
from typing import Callable
from pandas import DataFrame

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.bound_checker import bound_checker_builder
from corerl.data_pipeline.outlier_detectors.factory import init_outlier_detector
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.transition_creators.factory import init_transition_creator
from corerl.data_pipeline.state_constructors.factory import init_state_constructor

from corerl.data_pipeline.datatypes import Transition

from corerl.data_pipeline.pipeline_utils import warmup_pruning, handle_data_gaps

WARMUP = 0

type PipelineStage[T] = Callable[[T, str], T]
type WarmupPruner = Callable[[list[Transition], int], list[Transition]]

def invoke_stage_per_tag[T](carry: T, stage: Mapping[str, PipelineStage[T]]):
    for tag, f in stage.items():
        carry = f(carry, tag)

    return carry

class Pipeline:
    def __init__(self, cfg: MainConfig):
        self.missing_data = {
            cfg.name: missing_data_checker for cfg in cfg.tags
        }

        self.bound_checker = {
            cfg.name: bound_checker_builder(cfg) for cfg in cfg.tags
        }

        self.transition_creator = init_transition_creator(cfg.agent_transition_creator)

        self.outlier_detector = {
            cfg.name: init_outlier_detector(cfg.outlier) for cfg in cfg.tags
        }

        self.imputer = {
            cfg.name: init_imputer(cfg.imputer, cfg) for cfg in cfg.tags
        }

        self.state_constructor = {
            cfg.name: init_state_constructor(cfg.state_constructor) for cfg in cfg.tags
        }

        self.warmup_pruning: WarmupPruner = warmup_pruning

    def __call__(self, data: DataFrame) -> list[Transition]:
        pf = PipelineFrame(data)
        pf = invoke_stage_per_tag(pf, self.missing_data)
        pf = invoke_stage_per_tag(pf, self.bound_checker)
        pf = invoke_stage_per_tag(pf, self.outlier_detector)
        pf = invoke_stage_per_tag(pf, self.imputer)
        pfs = handle_data_gaps(pf)
        transitions = []
        for pf in pfs:
            pf_transitions = self.transition_creator(pf)
            pf_transitions = invoke_stage_per_tag(pf_transitions, self.state_constructor)
            pf_transitions = self.warmup_pruning(pf_transitions, WARMUP)  # placeholder for WARMUP. Comes from somewhere
            transitions += pf_transitions

        return transitions
