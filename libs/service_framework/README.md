# Service Framework

Base service framework for RLTune microservices, providing consistent lifecycle management, event bus integration, and signal handling.

## Features

- **Lifecycle Management**: Standardized start/stop/run lifecycle with state tracking
- **Event Bus Integration**: Automatic setup and teardown of event bus clients
- **Signal Handling**: Proper SIGTERM (graceful) and SIGINT (immediate) handling
- **Retry Logic**: Exponential backoff with configurable retry windows
- **State Queries**: Methods to check service state (running, stopped, failed)

## Usage

```python
from service_framework import RLTuneService, ServiceState
from lib_defs.type_defs.base_events import EventTopic

class MyService(RLTuneService):
    def __init__(self, config):
        super().__init__(
            service_name="my_service",
            event_topic=EventTopic.my_topic,
            event_bus_host=config.host,
            event_bus_pub_port=config.pub_port,
            event_bus_sub_port=config.sub_port,
        )
        self.config = config

    async def _do_start(self) -> None:
        # Initialize resources
        pass

    async def _do_run(self) -> None:
        # Main service loop - include await asyncio.sleep(0) for cancellation
        while True:
            # Do work
            await asyncio.sleep(0)  # Critical for signal handling!

    async def _do_stop(self) -> None:
        # Cleanup resources
        pass

# Run the service
service = MyService(config)
service.run_forever(
    max_retries=5,
    retry_window_hours=1,
    enable_retry=True,
)
```

## Important Notes

- **Cancellation Points**: Your `_do_run()` implementation must include `await` statements to allow task cancellation. Use `await asyncio.sleep(0)` in loops to yield control to the event loop.
- **Signal Handling**: SIGTERM triggers graceful shutdown (calls `_do_stop()`), SIGINT triggers immediate cancellation (skips `_do_stop()`).
- **Event Bus**: The event bus client is automatically set up and torn down. Access it via `self.get_event_bus_client()`.
