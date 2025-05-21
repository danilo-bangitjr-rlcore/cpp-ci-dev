from corerl.configs.config import MISSING, config, list_


@config()
class OPCConnectionConfig():
    connection_id: str = MISSING
    opc_conn_url: str = MISSING
    client_cert_path: str | None = None
    client_private_key_path: str | None = None
    server_cert_path: str | None = None
    application_uri: str | None = None

@config()
class CoreIOConfig():
    coreio_origin: str = "tcp://localhost:5557"
    opc_connections: list[OPCConnectionConfig] = list_()

