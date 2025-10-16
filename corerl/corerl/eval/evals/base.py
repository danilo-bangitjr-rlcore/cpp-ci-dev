import logging
from datetime import datetime
from typing import Protocol

import pandas as pd

log = logging.getLogger(__name__)


class EvalsWriterProtocol(Protocol):
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
        ...

    def flush(self) -> None:
        ...

    def close(self) -> object:
        ...
