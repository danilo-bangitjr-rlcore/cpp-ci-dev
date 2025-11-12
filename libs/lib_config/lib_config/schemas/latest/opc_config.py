from enum import StrEnum, auto
from pathlib import Path
from typing import Literal

from pydantic_extra_types.semantic_version import SemanticVersion

from lib_config.config import Config


# ------------------
# -- OPC Security --
# ------------------
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


class OPCSecurityPolicyBasic256SHA256Config(Config):
    policy: Literal[OPCSecurityPolicy.basic256_sha256] = OPCSecurityPolicy.basic256_sha256
    mode: OPCMessageSecurityMode = OPCMessageSecurityMode.sign_and_encrypt
    client_cert_path: Path
    client_key_path: Path
    server_cert_path: Path


class OPCAuthModelUsernamePasswordConfig(Config):
    username: str
    password: str = ""


# --------------------
# -- OPC Connection --
# --------------------
class OPCConnectionConfig(Config):
    connection_id: str
    client_timeout: int = 30  # in seconds
    client_watchdog_interval: int = 30  # in seconds
    connection_url: str
    application_uri: str | None = None
    security_policy: OPCSecurityPolicyBasic256SHA256Config | Literal[OPCSecurityPolicy.none]
    authentication_mode: OPCAuthModelUsernamePasswordConfig | Literal[OPCAuthMode.anonymous]


# ----------
# -- Tags --
# ----------
class OPCTagConfig(Config):
    name: str
    connection_id: str | None = None
    node_identifier: str | None = None


class OPCConfig(Config):
    schema_version: SemanticVersion = SemanticVersion(major=1, minor=0, patch=0)

    connections: list[OPCConnectionConfig] = []
    tags: list[OPCTagConfig] = []
