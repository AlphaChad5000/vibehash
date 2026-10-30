"""HTTPS for nodes: server contexts, self-signed testnet certificates, and certificate pinning for clients.

A node serves HTTPS itself with `vibehash node --tls-cert FILE --tls-key FILE`.
For a testnet, `vibehash tls cert` makes a self-signed certificate and prints
its SHA-256 fingerprint. Workers pin it with `setup --tls-fingerprint`, which
saves it next to the node's address (settings.py): from then on every command
accepts that exact certificate from that node and nothing else, with no
certificate authority involved. `--tls-ca FILE` instead trusts a private CA,
or the self-signed certificate itself, with the usual hostname checks. With
neither, an https node is checked against the system's CAs like any website.

Self-signed certificates need the `cryptography` package; without it, `tls
cert` prints the equivalent openssl command.
"""
from __future__ import annotations

import datetime
import hashlib
import http.client
import ipaddress
import os
import re
import shlex
import ssl
import urllib.request
from pathlib import Path


class TLSError(ValueError):
    pass


def server_context(cert: str | Path, key: str | Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        ctx.load_cert_chain(str(cert), str(key))
    except (OSError, ssl.SSLError) as e:
        raise TLSError(f"can't load the TLS certificate {cert} and key {key}: {e}") from e
    return ctx


def fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def cert_fingerprint(path: str | Path) -> str:
    """SHA-256 of a PEM certificate's DER bytes: what `openssl x509 -fingerprint -sha256` prints, without colons.
    For a chain (a CA's fullchain.pem: the site's certificate, then its issuers), the first one: the site's."""
    try:
        first = re.search(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", Path(path).read_text(), re.S)
        if first is None:
            raise ValueError("no certificate in it")
        return fingerprint(ssl.PEM_cert_to_DER_cert(first.group(0)))
    except ValueError as e:
        raise TLSError(f"{path} isn't a PEM certificate") from e


def normalize_fingerprint(text: str) -> str:
    """Accepts plain hex or openssl's 'SHA256 Fingerprint=AB:CD:...'."""
    fp = re.sub(r"[\s:]", "", text.split("=")[-1]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", fp):
        raise TLSError("a certificate fingerprint is 64 hex digits (SHA-256), optionally separated by colons")
    return fp


def default_hosts() -> list[str]:
    import socket

    hosts = ["localhost", "127.0.0.1", socket.gethostname()]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # picks the LAN interface; sends nothing
            hosts.append(s.getsockname()[0])
    except OSError:
        pass
    return list(dict.fromkeys(hosts))


def openssl_command(cert: str | Path, key: str | Path, hosts: list[str], days: int = 365) -> str:
    san = ",".join(f"IP:{h}" if _is_ip(h) else f"DNS:{h}" for h in hosts)
    return " ".join(["openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes", f"-days {days}",
                     f"-keyout {shlex.quote(str(key))} -out {shlex.quote(str(cert))}", "-subj /CN=vibehash-node",
                     f"-addext subjectAltName={san}", "-addext keyUsage=critical,digitalSignature,keyCertSign",
                     "-addext extendedKeyUsage=serverAuth"])


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def can_make_certs() -> bool:
    try:
        from cryptography import x509  # noqa: F401
    except ImportError:
        return False
    return True


def make_self_signed(cert: str | Path, key: str | Path, hosts: list[str], days: int = 365) -> str:
    """Write a self-signed ECDSA P-256 certificate for `hosts` and its unencrypted key. Returns the fingerprint.
    It is its own CA (so it also works as a --tls-ca file) and has the extensions strict verifiers require."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as e:
        raise TLSError("making a certificate needs the `cryptography` package") from e
    cert, key = Path(cert), Path(key)
    for p in (cert, key):
        if p.exists():
            raise TLSError(f"{p} already exists; refusing to overwrite it")
    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vibehash node")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.IPAddress(ipaddress.ip_address(h)) if _is_ip(h) else x509.DNSName(h) for h in hosts]
    built = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(private.public_key())
             .serial_number(x509.random_serial_number())
             .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=days))
             .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
             .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=False, content_commitment=False,
                                          key_encipherment=False, data_encipherment=False, key_agreement=False,
                                          encipher_only=False, decipher_only=False), critical=True)
             .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
             .add_extension(x509.SubjectAlternativeName(alt), critical=False)
             .add_extension(x509.SubjectKeyIdentifier.from_public_key(private.public_key()), critical=False)
             .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(private.public_key()), critical=False)
             .sign(private, hashes.SHA256()))
    key.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_bytes(built.public_bytes(serialization.Encoding.PEM))
    return fingerprint(built.public_bytes(serialization.Encoding.DER))


# --- clients ----------------------------------------------------------------------------------------

class _PinnedConnection(http.client.HTTPSConnection):
    """Checks the server's certificate against the pinned fingerprint before a byte of the request is sent."""
    pin = ""

    def connect(self):
        super().connect()
        got = fingerprint(self.sock.getpeercert(binary_form=True) or b"")
        if got != self.pin:
            self.close()
            raise ssl.SSLCertVerificationError(1, f"the node's certificate isn't the pinned one (it has sha256 {got}, "
                                                  f"expected {self.pin}); refusing to talk to it")


class _PinnedHandler(urllib.request.HTTPSHandler):
    def __init__(self, context: ssl.SSLContext, pin: str):
        super().__init__(context=context)
        self._ctx, self._pin = context, pin

    def https_open(self, req):
        return self.do_open(self._connection, req)

    def _connection(self, host, **kw):
        conn = _PinnedConnection(host, context=self._ctx, **kw)
        conn.pin = self._pin
        return conn


def opener(ca_file: str | None = None, pin: str | None = None) -> urllib.request.OpenerDirector:
    """An opener for https URLs that trusts `ca_file` (with hostname checks) and/or exactly the certificate with
    SHA-256 fingerprint `pin`. A pin alone needs no CA: the fingerprint is the trust."""
    if ca_file:
        try:
            ctx = ssl.create_default_context(cafile=str(ca_file))
        except (OSError, ssl.SSLError) as e:
            raise TLSError(f"can't load the CA file {ca_file}: {e}") from e
    else:
        ctx = ssl.create_default_context()
    if not pin:
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    if not ca_file:
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    return urllib.request.build_opener(_PinnedHandler(ctx, normalize_fingerprint(pin)))


def opener_for(url: str, ca_file: str | None = None, pin: str | None = None) -> urllib.request.OpenerDirector | None:
    """What NodeClient opens `url` with: None (plain urllib) for http and for https checked against the system's
    CAs; otherwise an opener for the given trust, or the one saved for this node by `setup`."""
    if not url.lower().startswith("https://"):
        return None
    if ca_file is None and pin is None:
        from . import settings

        ca_file, pin = settings.tls_trust(url)
    return opener(ca_file, pin) if ca_file or pin else None
