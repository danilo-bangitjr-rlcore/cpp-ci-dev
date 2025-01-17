from corerl.configs.config import MISSING, config


@config()
class MessageBusConfig():
    enabled: bool = False
    scheduler_connection: str = "tcp://localhost:5555"
    cli_connection: str = "tcp://localhost:5556"
