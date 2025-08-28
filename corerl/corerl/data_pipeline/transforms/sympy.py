import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms import SympyConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.utils.sympy import to_sympy


class SympyTransform:
    def __init__(self, cfg: SympyConfig):
        self._cfg = cfg
        self._expression_str = cfg.expression
        self._sympy_expr, self._lambda_func, self._tag_names = to_sympy(self._expression_str)

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

        return (
            np.nan
            if any(np.isnan(val) for val in tag_values)
            else self._lambda_func(*tag_values)
        )

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
