from dataclasses import dataclass

from corerl.eval.writer import MetricsWriter
from corerl.messages.client import WebsocketClient


@dataclass
class AppState:
    metrics: MetricsWriter
    event_bus: WebsocketClient
