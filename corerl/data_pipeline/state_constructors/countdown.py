from dataclasses import dataclass

from corerl.utils.hydra import interpolate


@dataclass
class CountdownConfig:
    action_period: int = interpolate('${action_period}')
    kind: str = 'one_hot'



@dataclass
class CountdownTS:
    clock: int
    last_action_tick: int


class CountdownAdder:
    def __init__(self, cfg: CountdownConfig):
        self._cfg = cfg
