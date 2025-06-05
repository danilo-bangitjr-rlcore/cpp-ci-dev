from typing import Literal

import pandas as pd

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class NukeConfig(BaseTransformConfig):
    name: Literal['nuke'] = 'nuke'


class Nuke:
    def __init__(self, cfg: NukeConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        carry.transform_data = pd.DataFrame()
        return carry, None

    def reset(self) -> None:
        pass


transform_group.dispatcher(Nuke)
