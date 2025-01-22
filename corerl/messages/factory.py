from corerl.configs.config import config


@config()
class EventBusConfig():
    enabled: bool = False
    cli_connection: str = "tcp://localhost:5555"
    app_connection: str = "inproc://corerl_app"

