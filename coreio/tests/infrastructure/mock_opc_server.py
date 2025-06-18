import pathlib

import pytest
from asyncua import Node, Server, ua
from asyncua.crypto.permission_rules import SimpleRoleRuleset
from asyncua.crypto.truststore import TrustStore
from asyncua.crypto.validator import CertificateValidator, CertificateValidatorOptions
from asyncua.server.user_managers import CertificateUserManager

from tests.infrastructure.mock_opc_certs import ServerClientKeyCerts


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
async def server(opc_port: int):
    s = FakeOpcServer(opc_port)
    await s.start()
    yield s
    await s.close()
