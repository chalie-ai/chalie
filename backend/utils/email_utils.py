from __future__ import annotations

import re

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)


class Email:

    @staticmethod
    def validate(address: str) -> bool:
        return bool(_EMAIL_RE.match(address.strip()))

    @staticmethod
    def get_domain(address: str) -> str:
        return address.rsplit("@", 1)[-1].strip().lower()

    @staticmethod
    def get_local(address: str) -> str:
        return address.rsplit("@", 1)[0].strip()

    @staticmethod
    def normalize(address: str) -> str:
        return address.strip().lower()
