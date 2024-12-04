from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography import x509
import cryptography.x509.oid as oid
import datetime
import ipaddress
from pathlib import Path
import argparse


def gen_cert_key_pair(target_path: Path, name: str = "test", server: bool = False):
    if server:
        key_agreement = True
        name = "rlcore-server"
    else:
        key_agreement = False

    # for opc servers, uri MUST start with urn:
    uri = f"urn:{name}"
    dns_name = "localhost"
    ip_address = "127.0.0.1"

    country_name = "CA"
    state_or_province_name = "Alberta"
    locality_name = "Edmonton"
    organization_name = "RLCore"

    # from https://cryptography.io/en/latest/x509/tutorial/#creating-a-self-signed-certificate

    # Generate our key

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write our key to disk for safe keeping

    with open(target_path / "key.pem", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            # encryption_algorithm=serialization.BestAvailableEncryption(b"passphrase"),
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Various details about who we are. For a self-signed certificate the
    # subject and issuer are always the same.

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, country_name),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state_or_province_name),
        x509.NameAttribute(NameOID.LOCALITY_NAME, locality_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization_name),
        x509.NameAttribute(NameOID.COMMON_NAME, dns_name),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.now(datetime.timezone.utc)
    ).not_valid_after(
        # Our certificate will be valid for 10 days
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1000)
    ).add_extension(
        x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
        critical=False,
    ).add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),
        critical=False,
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(dns_name),
            x509.UniformResourceIdentifier(uri),
            x509.IPAddress(ipaddress.ip_address(ip_address))
        ]),
        critical=False,
    # Sign our certificate with our private key
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=True,
            key_encipherment=True,
            data_encipherment=True,
            key_agreement=key_agreement,
            key_cert_sign=True,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=False,
    ).add_extension(
        x509.ExtendedKeyUsage([
            oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            oid.ExtendedKeyUsageOID.SERVER_AUTH,
        ]),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=False
    ).sign(key, hashes.SHA256())

    # Write our certificate out to disk.
    if server:
        with open(target_path / "cert.der", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.DER))
    else:
        with open(target_path / "cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        help="Generate keys for a server or a client (no-server)",
        required=True,
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    if args.server:
        # Server name must be "rlcore-server", hardcoded in function
        server = True
        gen_cert_key_pair(target_path=Path.cwd(), server=server)
    else:
        name = "test"
        server = False
        gen_cert_key_pair(target_path=Path.cwd(), name=name, server=server)
