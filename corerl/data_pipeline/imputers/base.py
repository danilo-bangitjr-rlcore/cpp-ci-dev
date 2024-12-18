import pandas as pd

from typing import Any, Dict, cast
from dataclasses import dataclass
from abc import abstractmethod

from corerl.configs.group import Group
from corerl.configs.config import config, MISSING
from corerl.data_pipeline.datatypes import StageCode, PipelineFrame


@config()
class BaseImputerConfig:
    name: Any = MISSING


@dataclass
class ImputerTemporalState:
    prev_pf_data: pd.DataFrame | pd.Series | None = None


class BaseImputer:
    def __init__(self, cfg: BaseImputerConfig):
        self.cfg = cfg

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
]()
