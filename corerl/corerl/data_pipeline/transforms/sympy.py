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

        result_values = []
        for idx in range(len(carry.transform_data)):
            tag_values = [carry.obs.iloc[idx][tag_name] for tag_name in self._tag_names]

            result = self._lambda_func(*tag_values)
            result_values.append(result)

        for col in transform_columns:
            carry.transform_data.drop(col, axis=1, inplace=True)

        carry.transform_data[carry.tag] = result_values
        return carry, None

    def reset(self) -> None:
        ...
