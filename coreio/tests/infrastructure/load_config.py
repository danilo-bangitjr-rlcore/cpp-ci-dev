from pathlib import Path

from lib_config.loader import direct_load_config

from coreio.communication.opc_communication import OPCConnectionConfig
from coreio.config import OPCSecurityPolicyBasic256SHA256Config
from coreio.utils.config_schemas import MainConfigAdapter
from tests.infrastructure.mock_opc_certs import ServerClientKeyCerts

def load_config(
        cfg_name: Path, opc_port: int, server_client_key_certs: ServerClientKeyCerts | None = None,
    ) -> OPCConnectionConfig:
    """
    Loads an agent config file, this should have CoreIO config in it
    for testing behavior. Replaces the opc_conn_url with a generated one
    for testing, and also replaces any key/cert info with generated ones
    """
    cfg = direct_load_config(
        MainConfigAdapter,
        config_name=str('tests/opc/' / cfg_name),
    )
    assert isinstance(cfg, MainConfigAdapter)
    config = cfg.coreio.opc_connections[0]
    if isinstance(config.security_policy, OPCSecurityPolicyBasic256SHA256Config):
        assert server_client_key_certs is not None, "Key certs must be provided if security policy is not None"

        return OPCConnectionConfig(
            connection_id=config.connection_id,
            application_uri=config.application_uri,
            authentication_mode=config.authentication_mode,
            opc_conn_url=f'opc.tcp://localhost:{opc_port}',
            security_policy=OPCSecurityPolicyBasic256SHA256Config(
                mode=config.security_policy.mode,
                client_cert_path=Path(server_client_key_certs.client.cert),
                client_key_path=Path(server_client_key_certs.client.key),
                server_cert_path=Path(server_client_key_certs.server.cert),
            ),
        )

    return OPCConnectionConfig(
        connection_id=config.connection_id,
        application_uri=config.application_uri,
        authentication_mode=config.authentication_mode,
        opc_conn_url=f'opc.tcp://localhost:{opc_port}',
        security_policy=config.security_policy,
    )

