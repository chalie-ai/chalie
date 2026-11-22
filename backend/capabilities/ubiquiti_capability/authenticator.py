"""The codes an authenticator app shows, made from the setup key a UI.com account gave it.

A code is the RFC 6238 one every authenticator app makes: SHA1, 6 digits, a new one every 30 seconds.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import struct
from urllib.parse import parse_qs, urlsplit

_DIGITS = 6
_STEP = 30
_LINK_SCHEME = "otpauth://"


def setup_key(raw: str) -> str:
    """The setup key as typed or pasted, tidied into padded upper-case base32.

    UI.com shows the key in groups of letters, so spaces go and case is ignored.
    A pasted ``otpauth://`` link gives its ``secret``; a link for any code other
    than the 6-digit, 30-second SHA1 one is refused.

    Raises:
        ValueError: the key is not base32, or the link is not for these codes.
            The message never contains the key.
    """
    text = raw.strip()
    if text[: len(_LINK_SCHEME)].lower() == _LINK_SCHEME:
        link = urlsplit(text)
        query = parse_qs(link.query)
        if (
            link.netloc.lower() != "totp"
            or query.get("algorithm", ["SHA1"])[0].upper() != "SHA1"
            or query.get("digits", [str(_DIGITS)])[0] != str(_DIGITS)
            or query.get("period", [str(_STEP)])[0] != str(_STEP)
        ):
            raise ValueError("The authenticator link is not for 6-digit codes that change every 30 seconds")
        text = query.get("secret", [""])[0]
    key = "".join(text.split()).upper().rstrip("=")
    key += "=" * (-len(key) % 8)
    try:
        decoded = base64.b32decode(key)
    except binascii.Error:
        decoded = b""
    if not decoded:
        raise ValueError("The authenticator setup key is not a valid key")
    return key


def code(key: str, at: float) -> str:
    """The code an authenticator app shows for a tidied setup key at the moment ``at`` (Unix seconds)."""
    mac = hmac.new(base64.b32decode(key), struct.pack(">Q", int(at // _STEP)), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    value = struct.unpack(">I", mac[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(value % 10**_DIGITS).zfill(_DIGITS)
