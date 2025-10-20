"""Sympy transform configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lib_config.config import MISSING, config, post_processor

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class SympyConfig(BaseTransformConfig):
    name: Literal["sympy"] = "sympy"
    expression: str = MISSING
    tolerance: float = 1e-4  # Tolerance for division operations, similar to inverse transform

    @post_processor
    def _validate_expression(self, cfg: MainConfig):
        from corerl.utils.sympy import is_expression, is_valid_expression, to_sympy

        # Validate sympy expression format and supported operations
        if not is_expression(self.expression):
            raise ValueError(f"Invalid sympy expression format: {self.expression}")

        expr, _, _ = to_sympy(self.expression)
        if not is_valid_expression(expr):
            raise ValueError(f"Expression contains unsupported operations: {self.expression}")
