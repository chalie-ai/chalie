"""ThreadScope — the ``type`` → ProcessorConfig/channel resolution every threads route shares.

``type`` (the ConfigType identity — ``user``/``scheduled``/``discovery``) is the
only routing surface the frontend speaks; it resolves to a transcript channel
server-side so a client can address other configs (e.g. the scheduler) without
knowing channel names. ``turn_id`` is only unique PER CHANNEL, so every threads
read and write carries it, and resolving it in one place keeps the endpoint and
its actions from drifting on the default or on what an unknown type means.
"""

from __future__ import annotations

from flask import request

from configs.channels import ChannelConfig, ProcessorConfig
from exceptions import EndpointError


class ThreadScope:
    """One request's resolved thread scope: the config type, its ProcessorConfig
    and the transcript channel that config reads and writes.

    An unrecognised type is a client error, not a crash — it surfaces as the
    uniform 400 envelope rather than the raw ``ValueError`` ``config_for``
    raises.
    """

    DEFAULT = "user"
    """Config type assumed when the caller names none — the human chat surface."""

    def __init__(self, config_type: str) -> None:
        try:
            self.config: ProcessorConfig = ChannelConfig.config_for(config_type)
        except ValueError as exc:
            raise EndpointError("Invalid type") from exc
        self.type = config_type
        self.channel: str = self.config.channel

    @classmethod
    def from_query(cls) -> ThreadScope:
        """Resolve from the ``type`` query arg — every read and the interrupt."""
        return cls(request.args.get("type") or cls.DEFAULT)

    @classmethod
    def from_form(cls) -> ThreadScope:
        """Resolve from the ``type`` multipart field — the send."""
        return cls(request.form.get("type") or cls.DEFAULT)
