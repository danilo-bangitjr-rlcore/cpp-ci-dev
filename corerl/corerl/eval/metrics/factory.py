from corerl.configs.eval.metrics import MetricsDBConfig
from corerl.eval.metrics.base import MetricsWriterProtocol
from corerl.eval.metrics.dummy import DummyMetricsWriter
from corerl.eval.metrics.narrow import NarrowMetricsTable
from corerl.eval.metrics.wide import WideMetricsTable


def create_metrics_writer(cfg: MetricsDBConfig) -> MetricsWriterProtocol:
    if not cfg.enabled:
        return DummyMetricsWriter()
    if cfg.narrow_format:
        return NarrowMetricsTable(cfg)
    return WideMetricsTable(cfg)
