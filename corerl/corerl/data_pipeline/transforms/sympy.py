import numpy as np
import pandas as pd
import sympy as sy

from corerl.data_pipeline.transforms import SympyConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.utils.sympy import to_sympy


class SympyTransform:
    def __init__(self, cfg: SympyConfig):
        self._cfg = cfg
        self._expression_str = cfg.expression
        self._sympy_expr, self._lambda_func, self._tag_names = to_sympy(self._expression_str)
        self._tolerance = cfg.tolerance

    def __call__(self, carry: TransformCarry, ts: object | None):
        transform_columns = list(carry.transform_data.columns)

        # Process each column in transform_data
        result_data: dict[str, list[float]] = {}
        for column in transform_columns:
            result_values = [
                self._compute_row_result(carry, idx, column)
                for idx in range(len(carry.transform_data))
            ]
            result_data[column] = result_values

        # Replace transform_data with transformed results
        carry.transform_data = pd.DataFrame(result_data, index=carry.transform_data.index)
        return carry, None

    def _compute_row_result(self, carry: TransformCarry, idx: int, current_column: str) -> float:
        tag_values = [
            self._get_tag_value(carry, tag_name, idx, current_column)
            for tag_name in self._tag_names
        ]

        if any(np.isnan(val) for val in tag_values):
            return np.nan

        # Check for division by zero with tolerance for division operations
        if self._has_division() and self._would_cause_division_by_zero(tag_values):
            return np.nan

        try:
            result = self._lambda_func(*tag_values)
        except (ZeroDivisionError, ValueError):
            # Handle division by zero and other math domain errors
            return np.nan

        # Convert boolean results to float (0.0/1.0) to match original transform behavior
        if isinstance(result, bool | np.bool_):
            return float(result)

        return result

    def _has_division(self) -> bool:
        """Check if the expression contains division operations."""
        # Check for division patterns in the sympy expression
        return any(
            hasattr(arg, 'is_Pow') and arg.is_Pow and
            len(arg.args) > 1 and arg.args[1].is_negative
            for arg in sy.preorder_traversal(self._sympy_expr)
        ) or any(
            hasattr(arg, 'is_Mul') and arg.is_Mul and
            any(hasattr(subarg, 'is_Pow') and subarg.is_Pow and
                len(subarg.args) > 1 and subarg.args[1].is_negative
                for subarg in arg.args)
            for arg in sy.preorder_traversal(self._sympy_expr)
        )

    def _would_cause_division_by_zero(self, tag_values: list[float]) -> bool:
        """Check if any tag values are too small and would cause division issues."""
        # For division operations, check if any denominators are too small
        # This is a simplified check - for more complex expressions, we might need
        # more sophisticated analysis
        return any(abs(val) <= self._tolerance for val in tag_values)

    def _get_tag_value(
        self,
        carry: TransformCarry,
        tag_name: str,
        idx: int,
        current_column: str,
    ) -> float:
        if tag_name == carry.tag:
            # Use the current column being processed from transform_data
            col_idx = carry.transform_data.columns.get_loc(current_column)
            return carry.transform_data.iat[idx, col_idx]

        # Get from original observations for all other tags
        col_idx = carry.obs.columns.get_loc(tag_name)
        return carry.obs.iat[idx, col_idx]

    def reset(self) -> None:
        ...
