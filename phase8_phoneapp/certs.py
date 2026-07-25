"""A self-signed certificate, so the phone will hand over its camera.

Browsers only expose `getUserMedia` (the camera) on a **secure context**: HTTPS, or
localhost. `http://192.168.0.14:8000` is neither, so the phone-as-camera app cannot exist
over plain HTTP no matter how the JavaScript is written. There is no way around this and
no flag worth asking a supervisor to set on their own phone.

So the server generates its own certificate. The phone will show a warning the first time
("your connection is not private") because nobody signed it — that is expected, and the
README says so plainly. Once it is accepted, the origin *is* a secure context and the
camera works.

To make accepting that warning an informed act rather than a superstition, `fingerprint()`
prints the certificate's SHA-256, and the phone's warning screen shows the same value
under "details". If they match, the warning is only saying "self-signed", not "someone is
in the middle".

The certificate is cached and reused, and regenerated when it expires or when the laptop's
IP changes (a different site, a different WiFi) — otherwise the address in the link would
not be one the certificate covers, and some browsers refuse that outright rather than
offering to proceed.
"""
from __future__ import annotations

import datetime
import ipaddress
import os
import socket
import subprocess

VALID_DAYS = 397          # Safari/Chrome reject leaf certs valid for much longer
RENEW_WITHIN_DAYS = 14


def local_addresses() -> list[str]:
    """Every IPv4 address this machine might be reached on.

    A laptop at a site is usually on two at once (WiFi and a phone hotspot), and the one
    the supervisor types is not always the one the routing table prefers. Covering all of
    them means the certificate stays valid when the laptop moves between networks, so the
    installed app keeps working instead of failing at the TLS layer.
    """
    found: list[str] = []

    def add(ip: str) -> None:
        if ip and ip not in found and not ip.startswith("169.254."):
            found.append(ip)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))        # no packet is sent; this just picks a route
        add(s.getsockname()[0])
    except OSError:
        pass
    finally:
        s.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass

    add("127.0.0.1")
    return found


def _covers(cert_path: str, hosts: list[str]) -> bool:
    """True if an existing certificate is still usable for every address in `hosts`."""
    try:
        from cryptography import x509
    except ImportError:
        return False
    try:
        with open(cert_path, "rb") as fh:
            cert = x509.load_pem_x509_certificate(fh.read())
        expires = cert.not_valid_after_utc
    except Exception:                                    # noqa: BLE001  unreadable/corrupt
        return False
    now = datetime.datetime.now(datetime.timezone.utc)
    if expires - now < datetime.timedelta(days=RENEW_WITHIN_DAYS):
        return False
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        names = {str(v) for v in san.get_values_for_type(x509.IPAddress)}
        names |= set(san.get_values_for_type(x509.DNSName))
    except x509.ExtensionNotFound:
        return False
    return all(h in names for h in hosts)


def _generate_cryptography(cert_path: str, key_path: str, hosts: list[str]) -> bool:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return False

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "AR Safety Monitor"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Site safety demo (self-signed)"),
    ])
    alt: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for h in hosts:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            alt.append(x509.DNSName(h))

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            # Backdate slightly: a phone whose clock is a few minutes behind the laptop
            # would otherwise reject a brand-new certificate as "not yet valid", which
            # looks like a broken app rather than a clock problem.
            .not_valid_before(now - datetime.timedelta(hours=1))
            .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    with open(key_path, "wb") as fh:
        fh.write(key.private_bytes(serialization.Encoding.PEM,
                                   serialization.PrivateFormat.TraditionalOpenSSL,
                                   serialization.NoEncryption()))
    with open(cert_path, "wb") as fh:
        fh.write(cert.public_bytes(serialization.Encoding.PEM))
    _restrict(key_path)
    return True


def _generate_openssl(cert_path: str, key_path: str, hosts: list[str]) -> bool:
    """Fallback for an environment without `cryptography` but with the openssl binary."""
    san = ",".join(
        (f"IP:{h}" if _is_ip(h) else f"DNS:{h}") for h in hosts) + ",DNS:localhost"
    cmd = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
           "-keyout", key_path, "-out", cert_path, "-days", str(VALID_DAYS),
           "-subj", "/CN=AR Safety Monitor", "-addext", f"subjectAltName={san}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    _restrict(key_path)
    return os.path.isfile(cert_path) and os.path.isfile(key_path)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _restrict(path: str) -> None:
    """Best effort: the private key should not be world-readable. Windows ignores the
    POSIX mode, which is why this is best effort and not a security control."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def ensure_cert(cert_dir: str, hosts: list[str] | None = None) -> tuple[str, str] | None:
    """Return (cert_path, key_path), generating them if needed. None if impossible."""
    hosts = hosts or local_addresses()
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "server.crt")
    key_path = os.path.join(cert_dir, "server.key")

    if (os.path.isfile(cert_path) and os.path.isfile(key_path)
            and _covers(cert_path, hosts)):
        return cert_path, key_path

    if _generate_cryptography(cert_path, key_path, hosts):
        return cert_path, key_path
    if _generate_openssl(cert_path, key_path, hosts):
        return cert_path, key_path
    return None


def fingerprint(cert_path: str) -> str:
    """SHA-256 of the certificate, formatted like the browser shows it.

    Printing this is the difference between "ignore the warning" and "check that the
    warning is only about self-signing": the phone shows the same string under the
    warning's details.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        with open(cert_path, "rb") as fh:
            cert = x509.load_pem_x509_certificate(fh.read())
        raw = cert.fingerprint(hashes.SHA256())
    except Exception:                                    # noqa: BLE001
        try:
            import hashlib
            import ssl
            with open(cert_path, "r", encoding="utf-8") as fh:
                der = ssl.PEM_cert_to_DER_cert(fh.read())
            raw = hashlib.sha256(der).digest()
        except Exception:                                # noqa: BLE001
            return "(unavailable)"
    return ":".join(f"{b:02X}" for b in raw)
