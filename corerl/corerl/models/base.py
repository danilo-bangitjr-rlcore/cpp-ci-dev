from typing import Any

from torch import Tensor

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.tag_config import TagConfig, TagType


@config()
class BaseModelConfig:
    name: Any = MISSING
    endogenous: bool = False


class BaseModel:
    def __init__(self,
                 cfg: BaseModelConfig,
                 tag_configs: list[TagConfig],
                 ):
        self.cfg = cfg
        self.endogenous = cfg.endogenous
        self._init_tag_info(tag_configs)

    def _init_tag_info(self, tag_configs: list[TagConfig]):
        self.endo_tags = []
        self.exo_tags = []
        for tag_config in tag_configs:
            name = tag_config.name
            if tag_config.type == TagType.ai_setpoint:
                pass
            elif name in ['reward', 'trunc', 'term']:
                pass
            elif tag_config.is_endogenous:
                self.endo_tags.append(name)
            else:
                self.exo_tags.append(name)

        self.endo_tags.sort()
        self.exo_tags.sort()


        start_endo_idxs = 0
        num_endo_idxs = len(self.endo_tags)
        self.endo_idxs = list(range(start_endo_idxs, start_endo_idxs + num_endo_idxs))

        start_exo_idxs = num_endo_idxs
        num_exo_idxs = len(self.exo_tags)
        self.exo_idxs = list(range(start_exo_idxs, start_exo_idxs + num_exo_idxs))


    def fit(self, transitions: list[Transition]):
        ...

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, float]:
        ...


model_group = Group[
    [list[TagConfig]], BaseModel
]()
