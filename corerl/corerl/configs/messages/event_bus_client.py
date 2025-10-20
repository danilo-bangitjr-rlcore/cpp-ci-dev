from lib_config.config import config


@config()
class EventBusClientConfig:
    enabled: bool = False
    host: str = "localhost"
    pub_port: int = 5559
    sub_port: int = 5560
