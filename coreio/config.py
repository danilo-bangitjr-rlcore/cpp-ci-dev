from corerl.configs.config import MISSING, config


@config()
class CoreIOConfig():
    opc_conn_url: str = MISSING
    # opc_ns: int | None = None  # OPC node namespace, this is almost always going to be `2`
    client_cert_path: str | None = None
    client_private_key_path: str | None = None
    server_cert_path: str | None = None
    application_uri: str | None = None
    coreio_connection: str = "tcp://localhost:5557"

