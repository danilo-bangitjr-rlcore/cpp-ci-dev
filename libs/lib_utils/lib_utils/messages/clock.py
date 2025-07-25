from datetime import UTC, datetime, timedelta
from typing import Generic

import zmq
from lib_defs.type_defs.base_events import EventClass, EventTopicClass, EventTypeClass

from lib_utils.messages.base_event_bus import BaseEventBus


class Clock(Generic[EventClass, EventTopicClass, EventTypeClass]): # noqa: UP046
    def __init__(
            self,
            event_class: type[EventClass],
            event_topic: EventTopicClass,
            event_type: EventTypeClass,
            period: timedelta,
            offset: timedelta = timedelta(seconds=0),
    ):
        self._event_class = event_class
        self._event_type = event_type
        self._event_topic = event_topic
        self._period = period

        self._next_ts = datetime.now(UTC) + offset

    def emit(self, event_bus: BaseEventBus[EventClass, EventTopicClass, EventTypeClass], now: datetime):
        event = self._event_class(type=self._event_type)
        try:
            event_bus.emit_event(event, topic=self._event_topic)
        except zmq.ZMQError as e:
            if isinstance(e, zmq.error.Again):
                # temporarily unavailable, retry
                return
            raise

        self.reset(now)

    def should_emit(self, now: datetime):
        return now > self._next_ts

    def maybe_emit(self, event_bus: BaseEventBus[EventClass, EventTopicClass, EventTypeClass], now: datetime):
        if self.should_emit(now):
            self.emit(event_bus, now)

    def get_next_ts(self):
        return self._next_ts

    def reset(self, now: datetime):
        self._next_ts = now + self._period
