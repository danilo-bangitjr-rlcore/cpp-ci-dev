from enum import StrEnum, auto
from pathlib import Path
from typing import Literal

from corerl.configs.config import MISSING, config, list_


class OPCSecurityPolicy(StrEnum):
    none = auto()
    basic256_sha256 = auto()

class OPCMessageSecurityMode(StrEnum):
    none = auto()
    sign = auto()
    sign_and_encrypt = auto()

class OPCAuthenticationMode(StrEnum):
    anonymous = auto()
    username_password = auto()

type OPCMessageSecurityModeNone = Literal[OPCMessageSecurityMode.none]
type OPCMessageSecurityModeNotNone = Literal[OPCMessageSecurityMode.sign, OPCMessageSecurityMode.sign_and_encrypt]
type OPCMessageSecurityModeAll = OPCMessageSecurityModeNone | OPCMessageSecurityModeNotNone

@config(frozen=True)
class OPCSecurityPolicyNone:
    policy: Literal[OPCSecurityPolicy.none] = OPCSecurityPolicy.none
    mode: OPCMessageSecurityModeNone = OPCMessageSecurityMode.none

@config(frozen=True)
class OPCSecurityPolicyBasic256SHA256:
    policy: Literal[OPCSecurityPolicy.basic256_sha256] = OPCSecurityPolicy.basic256_sha256
    mode: OPCMessageSecurityModeNotNone = OPCMessageSecurityMode.sign_and_encrypt
    client_cert_path: Path = MISSING
    client_key_path: Path = MISSING
    server_cert_path: Path = MISSING

type OPCSecurityPolicies = OPCSecurityPolicyNone | OPCSecurityPolicyBasic256SHA256

@config(frozen=True)
class OPCAuthenticationModeUsernamePassword:
    username: str = MISSING
    password: str = ""

type OPCAuthenticationModes = Literal[OPCAuthenticationMode.anonymous] | OPCAuthenticationModeUsernamePassword

@config(frozen=True)
class OPCConnectionConfig:
    connection_id: str = MISSING
    opc_conn_url: str = MISSING
    application_uri: str | None = None
    security_policy: OPCSecurityPolicies = MISSING
    authentication_mode: OPCAuthenticationModes = MISSING

@config(frozen=True)
class CoreIOConfig:
    coreio_origin: str = "tcp://localhost:5557"
    opc_connections: list[OPCConnectionConfig] = list_()
