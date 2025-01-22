
from corerl.eval.writer import MetricsWriterProtocol
from corerl.messages.event_bus import EventBus


class AppState:
    def __init__(self, metrics: MetricsWriterProtocol, event_bus: EventBus):
        self.metrics = metrics
        self.event_bus = event_bus
        self.agent_step = 0

