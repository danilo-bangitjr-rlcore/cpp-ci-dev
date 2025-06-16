from abc import abstractmethod
from typing import Any

from lib_config.config import MISSING, config
from lib_config.group import Group

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from corerl.state import AppState


@config()
class BaseImputerStageConfig:
    name: Any = MISSING


class BaseImputer:
    def __init__(self, imputer_cfg: BaseImputerStageConfig, app_state: AppState, tag_cfgs: list[TagConfig]):
        self._imputer_cfg = imputer_cfg
        self._app_state = app_state
        self._tags = tag_cfgs

    @abstractmethod
    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ...


imputer_group = Group[
    [AppState, list[TagConfig]],
    BaseImputer,
]()
