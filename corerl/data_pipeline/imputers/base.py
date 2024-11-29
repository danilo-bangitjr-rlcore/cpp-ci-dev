import pandas as pd

from typing import Any, Dict, cast
from dataclasses import dataclass
from abc import abstractmethod
from omegaconf import MISSING

from corerl.utils.hydra import Group
from corerl.data_pipeline.datatypes import StageCode, PipelineFrame


@dataclass
class BaseImputerConfig:
    name: str = MISSING


@dataclass
class ImputerTemporalState:
    prev_pf_data: pd.DataFrame | pd.Series | None = None


class BaseImputer:
    def __init__(self, cfg: BaseImputerConfig, tag_cfg: Any):
        self.cfg = cfg
        self.stage_code = StageCode.IMPUTER

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        stage_ts = pf.temporal_state.get(self.stage_code)
        if stage_ts is None:
            pf.temporal_state[self.stage_code] = dict()

        stage_ts = pf.temporal_state[self.stage_code]
        assert isinstance(stage_ts, dict)
        stage_ts = cast(
            Dict[str, ImputerTemporalState],
            stage_ts
        )

        imputer_ts = stage_ts.get(tag)
        pf, imputer_ts = self._inner_call(pf, tag, imputer_ts)
        stage_ts[tag] = imputer_ts

        return pf

    @abstractmethod
    def _inner_call(self, pf: PipelineFrame, tag: str, imputer_ts: ImputerTemporalState | None) \
            -> tuple[PipelineFrame, ImputerTemporalState]:
        raise NotImplementedError


imputer_group = Group[
    [Any], BaseImputer
]('pipeline/imputer')
