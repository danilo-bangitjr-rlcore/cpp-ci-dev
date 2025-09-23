import logging

from lib_utils.messages.base_event_bus import BaseEventBus

logger = logging.getLogger(__name__)


class ZMQ_Communication(BaseEventBus):
    async def async_listen_forever(self):
        while True:
            event = self.recv_event()
            if event is None:
                continue

            for cb in self._callbacks[event.type]:
                await cb(event)

            yield event
