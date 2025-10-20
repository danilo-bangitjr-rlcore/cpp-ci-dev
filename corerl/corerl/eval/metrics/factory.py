from collections.abc import Callable
from datetime import datetime

from corerl.configs.eval.metrics import MetricsDBConfig
from corerl.eval.metrics.base import MetricsWriterProtocol
from corerl.eval.metrics.dummy import DummyMetricsWriter
from corerl.eval.metrics.narrow import NarrowMetricsTable
from corerl.eval.metrics.wide import WideMetricsTable


def create_metrics_writer(
    cfg: MetricsDBConfig,
    time_provider: Callable[[], datetime] | None = None,
) -> MetricsWriterProtocol:
    if not cfg.enabled:
        return DummyMetricsWriter(cfg, time_provider)
    if cfg.narrow_format:
        return NarrowMetricsTable(cfg, time_provider)
    return WideMetricsTable(cfg, time_provider)
