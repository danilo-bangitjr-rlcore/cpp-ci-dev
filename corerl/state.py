from corerl.eval.eval_writer import EvalWriterProtocol
from corerl.eval.metrics_writer import MetricsWriterProtocol
from corerl.messages.event_bus import EventBus


class AppState:
    def __init__(self, metrics_writer: MetricsWriterProtocol, eval_writer: EvalWriterProtocol, event_bus: EventBus):
        self.metrics_writer = metrics_writer
        self.eval_writer = eval_writer
        self.event_bus = event_bus
        self.agent_step = 0

