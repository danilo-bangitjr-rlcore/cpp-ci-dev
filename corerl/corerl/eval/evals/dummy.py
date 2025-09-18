from datetime import datetime

import pandas as pd


class DummyEvalsWriter:
    """No-op evals writer for when evals are disabled."""

    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None) -> None:
        ...

    def read(
        self,
        evaluator: str,
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
