from corerl.configs.data_pipeline.transforms.identity import IdentityConfig
from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


class Identity:
    def __init__(self, cfg: IdentityConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        return carry, None

    def reset(self) -> None:
        pass


transform_group.dispatcher(Identity)
