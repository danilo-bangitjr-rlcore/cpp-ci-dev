from collections.abc import Mapping
from datetime import datetime
from typing import Any, SupportsFloat

import pandas as pd


class DummyMetricsWriter:
    """No-op metrics writer for when metrics are disabled."""

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
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...
