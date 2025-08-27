from lib_config.config import config


@config(allow_extra=True, frozen=True)
class MainConfigAdapter:
    ...

@config(allow_extra=True, frozen=True)
class DBConfigAdapter:
    drivername: str
    username: str
    password: str
    ip: str
    db_name: str
    port: int
