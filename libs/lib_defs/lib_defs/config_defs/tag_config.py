from enum import StrEnum, auto


class TagType(StrEnum):
    ai_setpoint = auto()
    meta = auto()
    seasonal = auto()
    delta = auto()
    default = auto()
