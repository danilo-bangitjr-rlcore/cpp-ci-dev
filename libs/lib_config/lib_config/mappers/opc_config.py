from pydantic_extra_types.semantic_version import SemanticVersion

from lib_config.mappers.base import Mapper
from lib_config.schemas.latest.opc_config import (
    OPCAuthMode,
    OPCAuthModelUsernamePasswordConfig,
    OPCConfig,
    OPCConnectionConfig,
    OPCSecurityPolicy,
    OPCSecurityPolicyBasic256SHA256Config,
    OPCTagConfig,
)
from lib_config.schemas.v0.opc_config import (
    CoreIOConfig,
    OPCAuthModeConfig,
    OPCAuthModeUsernamePasswordConfig,
    OPCSecurityPolicyConfig,
    OPCSecurityPolicyNoneConfig,
)

opc_mapper = Mapper(OPCConfig)


@opc_mapper.register_transform(
    version_from=SemanticVersion(0, 0, 0),
    version_to=SemanticVersion(1, 0, 0),
)
def transform_v0_to_v1(cfg: CoreIOConfig) -> OPCConfig:
    def parse_security_policy(old_policy: OPCSecurityPolicyConfig):
        if isinstance(old_policy, OPCSecurityPolicyNoneConfig):
            return OPCSecurityPolicy.none

        return OPCSecurityPolicyBasic256SHA256Config(
            client_cert_path=old_policy.client_cert_path,
            client_key_path=old_policy.client_key_path,
            server_cert_path=old_policy.server_cert_path,
        )

    def parse_authentication_mode(old_auth: OPCAuthModeConfig):
        if isinstance(old_auth, OPCAuthModeUsernamePasswordConfig):
            return OPCAuthModelUsernamePasswordConfig(
                username=old_auth.username,
                password=old_auth.password,
            )

        return OPCAuthMode.anonymous


    connections = [
        OPCConnectionConfig(
            connection_id=conn.connection_id,
            client_timeout=conn.client_timeout,
            client_watchdog_interval=conn.client_watchdog_interval,
            connection_url=conn.opc_conn_url,
            application_uri=conn.application_uri,
            security_policy=parse_security_policy(conn.security_policy),
            authentication_mode=parse_authentication_mode(conn.authentication_mode),
        )
        for conn in cfg.opc_connections
    ]

    return OPCConfig(
        connections=connections,
        tags=[
            OPCTagConfig(
                name=tag.name,
                connection_id=tag.connection_id,
                node_identifier=tag.node_identifier,
            )
            for tag in cfg.tags
        ],
    )
