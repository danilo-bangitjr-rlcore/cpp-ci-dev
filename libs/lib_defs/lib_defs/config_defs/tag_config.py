from enum import StrEnum, auto


class TagType(StrEnum):
    ai_setpoint = auto()
    meta = auto()
    day_of_year = auto()
    day_of_week = auto()
    time_of_day = auto()
    delta = auto()
    default = auto()
