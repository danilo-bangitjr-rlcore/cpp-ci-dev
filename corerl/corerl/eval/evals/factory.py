from corerl.eval.evals.base import EvalDBConfig, EvalsWriterProtocol
from corerl.eval.evals.dummy import DummyEvalsWriter
from corerl.eval.evals.static import StaticEvalsTable


def create_evals_writer(cfg: EvalDBConfig) -> EvalsWriterProtocol:
    if not cfg.enabled:
        return DummyEvalsWriter()
    return StaticEvalsTable(cfg)
