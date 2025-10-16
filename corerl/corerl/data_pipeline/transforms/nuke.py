import pandas as pd

from corerl.configs.data_pipeline.transforms.nuke import NukeConfig
from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


class Nuke:
    def __init__(self, cfg: NukeConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        carry.transform_data = pd.DataFrame()
        return carry, None

    def reset(self) -> None:
        pass


transform_group.dispatcher(Nuke)
