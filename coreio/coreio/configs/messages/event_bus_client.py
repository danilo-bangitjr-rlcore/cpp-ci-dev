from lib_config.config import config


@config()
class EventBusClientConfig:
    enabled: bool = True
    host: str = "localhost"
    pub_port: int = 5570
    sub_port: int = 5571
