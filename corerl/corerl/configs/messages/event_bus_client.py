from lib_config.config import config


@config()
class EventBusClientConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 5580
