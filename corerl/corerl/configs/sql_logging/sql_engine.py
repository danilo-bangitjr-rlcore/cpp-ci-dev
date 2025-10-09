from __future__ import annotations

from typing import TYPE_CHECKING

from lib_config.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class SQLEngineConfig:
    drivername: str = MISSING
    username: str = MISSING
    password: str = MISSING
    ip: str = MISSING
    port: int = MISSING

    @computed('drivername')
    @classmethod
    def _drivername(cls, cfg: MainConfig):
        return cfg.infra.db.drivername

    @computed('username')
    @classmethod
    def _username(cls, cfg: MainConfig):
        return cfg.infra.db.username

    @computed('password')
    @classmethod
    def _password(cls, cfg: MainConfig):
        return cfg.infra.db.password

    @computed('ip')
    @classmethod
    def _ip(cls, cfg: MainConfig):
        return cfg.infra.db.ip

    @computed('port')
    @classmethod
    def _port(cls, cfg: MainConfig):
        return cfg.infra.db.port
