from dataclasses import dataclass

import numpy as np


@dataclass
class ActionBoundsConfig:
    delta: bool = False
    static_lo: list[float] | float = 0.0
    static_hi: list[float] | float = 1.0
    delta_lo: list[float] | float | None = None
    delta_hi: list[float] | float | None = None


class ActionBoundsComputer:
    """Computes action bounds based on configuration and previous action."""

    def __init__(self, config: ActionBoundsConfig, action_dim: int):
        self.config = config
        self.action_dim = action_dim

        # Convert scalar bounds to arrays
        self.static_lo = self._to_array(config.static_lo)
        self.static_hi = self._to_array(config.static_hi)

        if config.delta:
            if config.delta_lo is None or config.delta_hi is None:
                raise ValueError("delta_lo and delta_hi must be provided when delta=True")
            self.delta_lo = self._to_array(config.delta_lo)
            self.delta_hi = self._to_array(config.delta_hi)
        else:
            self.delta_lo = None
            self.delta_hi = None

    def _to_array(self, value: float | list[float]) -> np.ndarray:
        """Converts scalar or list to numpy array of appropriate dimension."""
        if isinstance(value, (int, float)):
            return np.full(self.action_dim, value, dtype=np.float32)
        return np.array(value, dtype=np.float32)

    def compute(self, previous_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes action bounds given the previous action."""
        if not self.config.delta:
            # Static bounds only
            return self.static_lo, self.static_hi

        # Delta-based bounds, clamped to static limits
        a_lo = np.clip(
            previous_action - self.delta_lo,
            self.static_lo,
            self.static_hi,
        )
        a_hi = np.clip(
            previous_action + self.delta_hi,
            self.static_lo,
            self.static_hi,
        )

        return a_lo, a_hi
