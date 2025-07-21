from enum import StrEnum, auto
from pathlib import Path
from typing import Literal

from lib_config.config import MISSING, config, list_
from pydantic import Field


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


@config(frozen=True)
class OPCSecurityPolicyNoneConfig:
    policy: Literal[OPCSecurityPolicy.none] = OPCSecurityPolicy.none
    mode: Literal[OPCMessageSecurityMode.none] = OPCMessageSecurityMode.none


@config(frozen=True)
class OPCSecurityPolicyBasic256SHA256Config:
    policy: Literal[OPCSecurityPolicy.basic256_sha256] = OPCSecurityPolicy.basic256_sha256
    mode: OPCSecurityMode = OPCMessageSecurityMode.sign_and_encrypt
    client_cert_path: Path = MISSING
    client_key_path: Path = MISSING
    server_cert_path: Path = MISSING

type OPCSecurityPolicyConfig = OPCSecurityPolicyNoneConfig | OPCSecurityPolicyBasic256SHA256Config

@config(frozen=True)
class OPCAuthModeUsernamePasswordConfig:
    username: str = MISSING
    password: str = ""

type OPCAuthModeConfig = Literal[OPCAuthMode.anonymous] | OPCAuthModeUsernamePasswordConfig

@config(frozen=True)
class OPCConnectionConfig:
    connection_id: str = MISSING
    opc_conn_url: str = MISSING
    application_uri: str = ""
    security_policy: OPCSecurityPolicyConfig = MISSING
    authentication_mode: OPCAuthModeConfig = MISSING

@config(frozen=True)
class DataIngressConfig:
    enabled: bool = False

@config(frozen=True)
class CoreIOConfig:
    data_ingress: DataIngressConfig = Field(default_factory=DataIngressConfig)
    coreio_origin: str = "tcp://localhost:5557"
    opc_connections: list[OPCConnectionConfig] = list_()
