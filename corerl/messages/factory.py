from corerl.configs.config import config


@config()
class EventBusConfig():
    enabled: bool = False
    scheduler_connection: str = "tcp://localhost:5555"
    cli_connection: str = "tcp://localhost:5556"
    app_connection: str = "tcp://localhost:5557"

