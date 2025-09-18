import logging

from lib_utils.messages.base_event_bus import BaseEventBus

logger = logging.getLogger(__name__)


class ZMQ_Communication(BaseEventBus):
    async def async_listen_forever(self):
        logger.debug("EventBus - Starting async listen forever loop")
        while True:
            logger.debug("EventBus - About to receive event from queue")
            event = self.recv_event()
            if event is None:
                logger.debug("EventBus - No event received, continuing loop")
                continue

            logger.debug(f"EventBus - Received event {event.type},"
                         f" processing {len(self._callbacks[event.type])} callbacks")
            for cb in self._callbacks[event.type]:
                logger.debug(f"EventBus - About to execute callback for event {event.type}")
                await cb(event)

            yield event
