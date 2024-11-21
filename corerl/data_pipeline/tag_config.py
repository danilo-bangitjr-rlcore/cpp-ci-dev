from dataclasses import dataclass
from omegaconf import MISSING


@dataclass  # Placeholder
class TagConfig:
    name: str = MISSING
    bounds: tuple[float | None, float | None] = (None, None)
