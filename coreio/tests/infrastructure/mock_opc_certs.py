import datetime
import ipaddress
import pathlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


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
