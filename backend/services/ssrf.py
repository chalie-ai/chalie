"""Single SSRF blocklist all consumers import — no second copy."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# The single SSRF blocklist. Any consumer that needs to gate an outbound fetch
# imports THIS tuple — never re-declares its own. Drift is structurally
# impossible because there is exactly one definition.
BLOCKED_NETS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),        # "this host" / unspecified
    ipaddress.ip_network("10.0.0.0/8"),       # RFC 1918 private
    ipaddress.ip_network("100.64.0.0/10"),    # carrier-grade NAT (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("169.254.0.0/16"),   # link-local / cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),    # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),   # RFC 1918 private
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
)


def resolve_and_check(hostname: str) -> tuple[bool, str]:
    if not hostname:
        return False, "Empty hostname"
    try:
        results = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False, f"DNS resolution failed: {hostname}"

    for info in results:
        ip = ipaddress.ip_address(info[4][0])
        for net in BLOCKED_NETS:
            if ip in net:
                return False, f"Blocked IP range: {ip} ({hostname})"
    return True, ""


def is_private_url(url: str) -> bool:
    """``True`` when *url*'s host resolves to a blocked/private range (or fails).

    Fail-closed: a URL with no hostname, or one whose DNS lookup fails, is
    treated as private so the caller refuses to fetch it.
    """
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    ok, _ = resolve_and_check(hostname)
    return not ok
