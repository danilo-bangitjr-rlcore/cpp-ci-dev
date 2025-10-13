
from typing import Literal

from lib_config.config import config, list_

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig


@config()
class TraceConfig(BaseTransformConfig):
    name: Literal['multi_trace'] = 'multi_trace'
    trace_values: list[float] = list_([0., 0.75, 0.9, 0.95])
    missing_tol: float = 0.25  # proportion of the trace that can be "missing"
