
from corerl.component.buffer.mixed_history import MixedHistoryBufferConfig
from corerl.state import AppState

BufferConfig = MixedHistoryBufferConfig
from corerl.component.buffer.base import buffer_group


def init_buffer(cfg: BufferConfig, app_state: AppState):
    return buffer_group.dispatch(cfg, app_state)
