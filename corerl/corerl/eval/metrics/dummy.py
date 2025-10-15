from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, SupportsFloat

import pandas as pd

if TYPE_CHECKING:
    from corerl.eval.metrics.base import MetricsDBConfig


class DummyMetricsWriter:
    """No-op metrics writer for when metrics are disabled."""

    def __init__(self, cfg: 'MetricsDBConfig', time_provider: Callable[[], datetime] | None = None):
        pass

    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None) -> None:
        ...

    def write_dict(
        self,
        values: Mapping[str, SupportsFloat | Mapping[str, Any]],
        agent_step: int,
        prefix: str = '',
    ) -> None:
        ...

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...
