import asyncio

#from cryptography import utils
import datetime
import ipaddress
import pathlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from asyncua import Node, Server, ua
from asyncua.crypto.permission_rules import SimpleRoleRuleset
from asyncua.crypto.truststore import TrustStore
from asyncua.crypto.validator import CertificateValidator, CertificateValidatorOptions
from asyncua.server.user_managers import CertificateUserManager
from corerl.config import MainConfig
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from lib_config.loader import direct_load_config
from test.infrastructure.networking import get_free_port

from coreio.config import OPCSecurityPolicyBasic256SHA256Config
from coreio.utils.opc_communication import OPC_Connection, OPCConnectionConfig


@dataclass
class KeyCertNames:
    key: str
    cert: str

@dataclass
class ServerClientKeyCerts:
    path: pathlib.Path
    server: KeyCertNames
    client: KeyCertNames

def key_cert(
        path: pathlib.Path, uri: str, key_name: str, cert_name: str, trust_path: pathlib.Path | None = None,
    ) -> KeyCertNames:
    """
    Generate SSL key+cert pairs for use with OPC Servers and Clients
    """
    # for opc servers, uri MUST start with urn:

    # from https://cryptography.io/en/latest/x509/tutorial/#creating-a-self-signed-certificate

    # Generate our key

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write our key to disk for safe keeping

    key_path = path / (key_name + ".pem")
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            #encryption_algorithm=serialization.BestAvailableEncryption(b"passphrase"),
            encryption_algorithm=serialization.NoEncryption(),
        ))



    # Various details about who we are. For a self-signed certificate the
    # subject and issuer are always the same.

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CA"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Alberta"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Edmonton"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RLCore"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject,
    ).issuer_name(
        issuer,
    ).public_key(
        key.public_key(),
    ).serial_number(
        x509.random_serial_number(),
    ).not_valid_before(
        datetime.datetime.now(datetime.UTC),
    ).not_valid_after(
        # Our certificate will be valid for 10 days
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1000),
    ).add_extension(
        x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
        critical=False,
    ).add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),
        critical=False,
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.UniformResourceIdentifier(uri),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]),
        critical=False,
    # Sign our certificate with our private key
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=True,
            key_encipherment=True,
            data_encipherment=True,
            key_agreement=True,
            key_cert_sign=True,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=False,
    ).add_extension(
        x509.ExtendedKeyUsage([
            ExtendedKeyUsageOID.CLIENT_AUTH,
            ExtendedKeyUsageOID.SERVER_AUTH,
        ]),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Write our certificate out to disk.
    cert_path = path / (cert_name + ".der")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.DER))

    if isinstance(trust_path, Path):
        trust_path = trust_path / (cert_name + ".der")
        with open(trust_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.DER))

    return KeyCertNames(key=str(key_path), cert=str(cert_path))


class FakeOpcServer:
    def __init__(self, port: int):
        self._port = port
        self._s: Server | None = None
        self._sensors: dict[str, Node] = {}

    async def start(self):
        self._s = Server()
        await self._s.init()

        self._s.set_endpoint(f'opc.tcp://localhost:{self._port}/opcua/')
        self._s.set_server_name('RLCore Test Server')
        idx = await self._s.register_namespace('http://rlcore.test.ai/opcua/')

        virtual_plc = await self._s.nodes.objects.add_object(idx, 'vPLC1')
        self._sensors = {
            'temp': await virtual_plc.add_variable(idx, 'temp', 0.),
            'pressure': await virtual_plc.add_variable(idx, 'pressure', 0.),
        }

        await self._s.start()

    async def start_encrypt(self, server_client_key_certs: ServerClientKeyCerts):
        """
        Starts the OPC Server using Basic256_SHA256 and sign and encrypt
        Pre-populate a single trusted client
        """
        # certificate user manager associates public key certificates with
        # different users
        # the name param is actually not inportant since the cert is the identity here but should be unique
        #
        # currently you need to manually add these add_admin commands for each unique user connecting
        cert_user_manager = CertificateUserManager()
        await cert_user_manager.add_admin(
            pathlib.Path(server_client_key_certs.client.cert),
            name="client",
        )

        self._s = Server(user_manager=cert_user_manager)
        await self._s.init()
        await self._s.set_application_uri("urn:server")
        self._s.set_endpoint(f"opc.tcp://localhost:{self._port}")

        # connecting clients must use Basic256Sha256 and SignAndEncrypt
        # SimpleRoleRuleset:
        # Admin -> read/write
        # User -> read only
        # Anonymous -> none
        self._s.set_security_policy(
            [ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt],
            permission_ruleset=SimpleRoleRuleset(),
        )

        # load the servers pub+private keys
        await self._s.load_certificate(server_client_key_certs.server.cert)
        await self._s.load_private_key(server_client_key_certs.server.key, password=None, format=".pem")
        # this can be disabled for debugging
        #server.disable_clock(True)
        self._s.set_server_name("RLCore OPC-UA Server")

        # trusted client certificates MUST go in this directory, they won't be valid otherwise
        trust_store = TrustStore([server_client_key_certs.path / "trusted" / "certs"], [])
        await trust_store.load()
        validator = CertificateValidator(
            options=CertificateValidatorOptions.TRUSTED_VALIDATION | CertificateValidatorOptions.PEER_CLIENT,
            trust_store=trust_store,
        )
        self._s.set_certificate_validator(validator=validator)

        idx = await self._s.register_namespace('http://rlcore.test.ai/opcua/')

        virtual_plc = await self._s.nodes.objects.add_object(idx, 'vPLC1')
        self._sensors = {
            'temp': await virtual_plc.add_variable(idx, 'temp', 0.),
            'pressure': await virtual_plc.add_variable(idx, 'pressure', 0.),
        }

        await self._s.start()




    async def step(self, v: float):
        for j, sensor in enumerate(self._sensors.values()):
            await sensor.write_value(float(v + j))

    async def close(self):
        assert self._s is not None
        await self._s.stop()


@pytest.fixture
def opc_port():
    """
    (Presumably) Gets a free port from localhost that the server can listen on
    instead of assuming any particular one will be free
    """
    return get_free_port('localhost')

@pytest.fixture
async def server(opc_port: int):
    s = FakeOpcServer(opc_port)
    await s.start()
    yield s
    await s.close()

@pytest.fixture
async def client():
    client = OPC_Connection()
    yield client
    await client.cleanup()

@pytest.fixture
def client_server_key_certs(tmp_path: pathlib.Path) -> ServerClientKeyCerts:
    """
    Generates the client and server key+cert pairs for testing the
    encrypted OPC communication mechanisms.
    Files are stored in a temporary directory using the pytest tmp_path fixture
    """
    server_certs_path = tmp_path / "server" / "certs"
    client_certs_path = tmp_path / "client" / "certs"
    trust_path = tmp_path / "trusted" / "certs"
    server_certs_path.mkdir(parents=True, exist_ok=True)
    client_certs_path.mkdir(parents=True, exist_ok=True)
    trust_path.mkdir(parents=True, exist_ok=True)
    server_key_cert = key_cert(server_certs_path, "urn:server", "key", "cert")
    # client cert is also added to the servers trust path
    client_key_cert = key_cert(client_certs_path, "urn:client", "key", "cert", trust_path=trust_path)
    return ServerClientKeyCerts(tmp_path, server_key_cert, client_key_cert)

@pytest.fixture
async def server_key_cert(opc_port: int, client_server_key_certs: ServerClientKeyCerts):
    s = FakeOpcServer(opc_port)
    await s.start_encrypt(client_server_key_certs)
    yield s
    await s.close()

def load_config(
        cfg_name: Path, opc_port: int, server_client_key_certs: ServerClientKeyCerts | None = None,
    ) -> OPCConnectionConfig:
    """
    Loads an agent config file, this should have CoreIO config in it
    for testing behavior. Replaces the opc_conn_url with a generated one
    for testing, and also replaces any key/cert info with generated ones
    """
    cfg = direct_load_config(
        MainConfig,
        config_name=str('tests/opc/' / cfg_name),
    )
    assert isinstance(cfg, MainConfig)
    config = cfg.coreio.opc_connections[0]
    if (
        isinstance(config.security_policy, OPCSecurityPolicyBasic256SHA256Config) and
        isinstance(server_client_key_certs, ServerClientKeyCerts)
    ):
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

async def test_connect1(server: FakeOpcServer, client: OPC_Connection, opc_port: int):
    """
    Client should be able to connect to a running server.
    """
    config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(config, [])
    await client.start()


async def test_connect2(client: OPC_Connection, opc_port: int):
    """
    Client should fail if a server is not running.

    James: future behavior; client should attempt to re-connect to the server
    continuously, and should not fail.
    """
    with pytest.raises(OSError):
        client_config = load_config(Path("assets", "basic.yaml"), opc_port)
        await client.init(client_config, [])
        await client.start()

async def test_disconnect1(server: FakeOpcServer, client: OPC_Connection, opc_port: int):
    """
    Client survives when a server goes offline after connection.
    Check this sequence:
      1. Client and server connect
      2. Server closes
      3. Server starts
      4. Client implicitly reconnects in the background
    """
    client_config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(client_config, [])
    await server.close()
    await asyncio.sleep(0.1)
    await server.start()
    await client.ensure_connected()

async def test_connect_encrypt(
        server_key_cert: FakeOpcServer, client: OPC_Connection,
        opc_port: int, client_server_key_certs: ServerClientKeyCerts,
    ):
    """
    Basic256_SHA256, sign and encrypt, username password client can connect
    to a server endpoint with same config
    """
    client_config = load_config(Path("assets", "sha256_se_up.yaml"), opc_port, client_server_key_certs)
    await client.init(client_config, [])
    await client.start()
