from enum import StrEnum, auto
from pathlib import Path
from typing import Literal

from pydantic import Field

from lib_config.config import Config, ConfigWithExtra


class TagConfigAdapter(Config):
    name: str
    connection_id: str | None = None
    node_identifier: str | None = None

class OPCSecurityPolicy(StrEnum):
    none = auto()
    basic256_sha256 = auto()

class OPCMessageSecurityMode(StrEnum):
    none = auto()
    sign = auto()
    sign_and_encrypt = auto()

class OPCAuthMode(StrEnum):
    anonymous = auto()
    username_password = auto()


type OPCSecurityMode = Literal[OPCMessageSecurityMode.sign, OPCMessageSecurityMode.sign_and_encrypt]


class OPCSecurityPolicyNoneConfig(Config):
    policy: Literal[OPCSecurityPolicy.none] = OPCSecurityPolicy.none
    mode: Literal[OPCMessageSecurityMode.none] = OPCMessageSecurityMode.none


class OPCSecurityPolicyBasic256SHA256Config(Config):
    policy: Literal[OPCSecurityPolicy.basic256_sha256] = OPCSecurityPolicy.basic256_sha256
    mode: OPCSecurityMode = OPCMessageSecurityMode.sign_and_encrypt
    client_cert_path: Path
    client_key_path: Path
    server_cert_path: Path

type OPCSecurityPolicyConfig = OPCSecurityPolicyNoneConfig | OPCSecurityPolicyBasic256SHA256Config

class OPCAuthModeUsernamePasswordConfig(Config):
    username: str
    password: str = ""

type OPCAuthModeConfig = Literal[OPCAuthMode.anonymous] | OPCAuthModeUsernamePasswordConfig

class OPCConnectionConfig(Config):
    connection_id: str
    client_timeout: int = 30
    client_watchdog_interval: int = 30
    opc_conn_url: str
    application_uri: str | None = None
    security_policy: OPCSecurityPolicyConfig
    authentication_mode: OPCAuthModeConfig


class CoreIOConfig(ConfigWithExtra):
    opc_connections: list[OPCConnectionConfig] = Field(default_factory=list)
    tags: list[TagConfigAdapter] = Field(default_factory=list)
