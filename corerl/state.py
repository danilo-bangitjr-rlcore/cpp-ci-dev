from corerl.eval.evals import EvalWriterProtocol
from corerl.eval.metrics import MetricsWriterProtocol
from corerl.messages.event_bus import EventBus


class AppState:
    def __init__(self, metrics: MetricsWriterProtocol, evals: EvalWriterProtocol, event_bus: EventBus):
        self.metrics = metrics
        self.evals = evals
        self.event_bus = event_bus
        self.agent_step = 0

