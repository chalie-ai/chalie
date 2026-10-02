"""ThreadScope — the ``type`` → ProcessorConfig/channel resolution every threads route shares.

``type`` (the ConfigType identity — ``user``/``scheduled``/``discovery``) is the
only routing surface the frontend speaks; it resolves to a transcript channel
server-side so a client can address other configs (e.g. the scheduler) without
knowing channel names. ``turn_id`` is only unique PER CHANNEL, so every threads
read and write carries it, and resolving it in one place keeps the endpoint and
its actions from drifting on the default or on what an unknown type means.

The single-turn read alone also accepts ``channel``: a watchable delegate turn
has no ConfigType of its own, so it is addressed by its channel instead.
"""

from __future__ import annotations

from flask import request

from configs.channels import ProcessorConfig, config_for
from configs.enums.channels import WATCHABLE_DELEGATE_CHANNELS
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
            self.config: ProcessorConfig = config_for(config_type)
        except ValueError as exc:
            raise EndpointError("Invalid type") from exc
        self.type = config_type
        self.channel: str = self.config.channel

    @classmethod
    def from_query(cls) -> ThreadScope:
        """Resolve from the ``type`` query arg — every read and the interrupt.
        Only the single-turn read is addressed by channel (:meth:`read_target`),
        so naming a ``channel`` here is refused rather than landing on the type's."""
        if "channel" in request.args:
            raise EndpointError("Only a single-turn read is addressed by channel")
        return cls(request.args.get("type") or cls.DEFAULT)

    @classmethod
    def from_form(cls) -> ThreadScope:
        """Resolve from the ``type`` multipart field — the send. Only a delegate
        turn is addressed by channel, and only its caller drives it, so a send
        naming a ``channel`` is refused rather than landing on the type's."""
        if "channel" in request.values:
            raise EndpointError("A send is addressed by type, not channel")
        return cls(request.form.get("type") or cls.DEFAULT)

    @classmethod
    def read_target(cls) -> tuple[str, str | None]:
        """Resolve the single-turn read to ``(channel, type)``: by ``type`` as
        every other route, or by a ``channel`` naming a watchable delegate turn,
        whose type is None. Naming both is ambiguous, so it is refused."""
        if "channel" not in request.args:
            scope = cls.from_query()
            return scope.channel, scope.type
        if "type" in request.args:
            raise EndpointError("Pass type or channel, not both")
        channel = request.args["channel"]
        if channel not in WATCHABLE_DELEGATE_CHANNELS:
            raise EndpointError("Invalid channel")
        return channel, None
