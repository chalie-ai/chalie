"""Email address utilities — validation, domain extraction, normalisation."""
from __future__ import annotations

import re


_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)


class Email:
    """Static helpers for email address handling."""

    @staticmethod
    def validate(address: str) -> bool:
        """Return ``True`` if *address* looks like a syntactically valid email."""
        return bool(_EMAIL_RE.match(address.strip()))

    @staticmethod
    def get_domain(address: str) -> str:
        """Extract and normalise the domain part of an email address."""
        return address.rsplit("@", 1)[-1].strip().lower()

    @staticmethod
    def get_local(address: str) -> str:
        """Extract the local-part (everything before ``@``)."""
        return address.rsplit("@", 1)[0].strip()

    @staticmethod
    def normalize(address: str) -> str:
        """Return a lower-cased, stripped copy of *address*."""
        return address.strip().lower()
