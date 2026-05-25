from abc import ABC, abstractmethod
from typing import ClassVar


class Ability(ABC):
    """Base class for every dispatchable tool.

    Tool *scope* (always-available vs discoverable) lives on the calling
    MessageProcessor — each subclass declares ``ALWAYS_AVAILABLE`` and
    ``DISCOVERABLE`` lists of tool names. Abilities themselves only describe
    what the tool is (NAME / SUMMARY / EXAMPLES / INPUT_SCHEMA / TIMEOUT).
    """

    NAME: ClassVar[str]
    SUMMARY: ClassVar[str]
    EXAMPLES: ClassVar[list[str]]
    INPUT_SCHEMA: ClassVar[dict]
    TIMEOUT: ClassVar[int] = 10
    INTERNAL: ClassVar[bool] = False
    SEARCH_TOOLTIP: ClassVar[str] = ""
    POLICY_CATEGORY: ClassVar[str] = ""
    POLICY_LABELS: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses (marked with abstractmethod) are exempt.
        if ABC in cls.__bases__:
            return
        # Skip intermediate abstract classes that still have abstract methods.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("NAME", "INPUT_SCHEMA"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        if getattr(cls, "INTERNAL", False):
            return
        for attr in ("SUMMARY", "EXAMPLES"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        if not isinstance(cls.EXAMPLES, list) or not all(
            isinstance(e, str) for e in cls.EXAMPLES
        ):
            raise TypeError(f"{cls.__name__}.EXAMPLES must be list[str]")
        if not (6 <= len(cls.EXAMPLES) <= 8):
            raise TypeError(
                f"{cls.__name__}.EXAMPLES must have 6–8 entries, got {len(cls.EXAMPLES)}"
            )
        if (
            getattr(cls, "__module__", "").startswith("abilities.")
            and not getattr(cls, "SEARCH_TOOLTIP", "")
        ):
            raise TypeError(f"{cls.__name__} must define a non-empty SEARCH_TOOLTIP")

    def pre_dispatch(self, params: dict) -> dict:
        """Called before policy enforcement; may return a modified params dict.

        Override to normalise or escalate parameters before ActDispatcher runs
        the policy check.  The default returns *params* unchanged.
        """
        return params

    def get_description(self) -> str:
        """Return the tool description for LLM tool presentation.

        Override to enrich the description at runtime (e.g. find_tools
        appends a discoverable-tools index).  The default returns SUMMARY.
        """
        return self.SUMMARY

    def get_input_schema(self) -> dict:
        """Return the INPUT_SCHEMA for LLM tool presentation.

        Override to enrich the schema at runtime.  The default returns
        the class-level INPUT_SCHEMA unchanged.
        """
        return self.INPUT_SCHEMA

    @staticmethod
    def handle(handler, params: dict, telemetry: dict | None) -> dict:
        action_params = {k: v for k, v in params.items() if not k.startswith("_") and k != "action"}
        return handler(topic="", params=action_params, telemetry=telemetry)

    @abstractmethod
    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        """Execute the ability and return a result.

        Ordinal-None contract
        ─────────────────────
        ``params`` may contain the key ``_rich_media_ordinal`` (an ``int``)
        injected by ``ActDispatcherService.dispatch_action()`` ONLY when the
        call originates from a ``channel == 'user'`` dispatch.  This is the
        single physical gate that controls whether the rich-media instruction
        trailer is appended to the tool result.

        When ``params`` does NOT contain ``_rich_media_ordinal`` (i.e. ordinal
        is ``None``) the ability MUST NOT emit a rich-media instruction trailer.
        It must return a plain ``dict`` (or a plain string with no trailer
        signature). The trailer is only appended when the dispatcher injects the
        ordinal, and that only happens on ``channel == 'user'``.

        Paths where ``_rich_media_ordinal`` is absent (never inject a trailer):

        * ``channel == 'subagent'`` — dispatcher withholds the ordinal;
          ``SubagentProcessor`` also scrubs spans via ``strip_spans()`` as a
          second line of defence.
        * Action-button handler (``api/websocket.py::_handle_action``) — the
          ability is invoked via the dispatcher with a synthetic
          ``'action_button'`` channel, which is not ``'user'``, so no ordinal
          is injected.
        * Direct test calls — unit tests call ``execute()`` without the ordinal
          key and must receive a plain dict result.

        Args:
            channel: The conversation channel for this invocation (e.g.
                ``'user'``, ``'subagent'``, ``'action_button'``).
            params: Input parameters from the LLM action dict (or action-button
                payload), with framework keys (``type``, ``exchange_id``) already
                stripped by the dispatcher.  May contain
                ``_rich_media_ordinal: int`` when called from a user-channel
                dispatch.
            telemetry: Optional client telemetry dict (location, device info),
                or ``None`` if not available.

        Returns:
            ``dict`` when the result is structured data (no rich-media trailer),
            or ``str`` when the dispatcher-injected ordinal is present and the
            ability supports rich-media rendering (the string encodes the JSON
            payload + ``\\n\\n`` + the instruction trailer).
        """
        ...

    @classmethod
    def enrich_rich_payload(cls, payload: dict, row: dict) -> dict:
        """Resolve a rich-media payload's runtime state at parse time.

        The default implementation returns the payload unchanged. Override on
        subclasses whose card needs data that does NOT live in the LLM-visible
        tool result — for example:

        * ``timer`` injects ``started_at`` from the row's ``created_at`` so the
          wall-clock anchor never reaches the LLM.
        * ``list`` re-fetches the live list from ``ListService`` so checkbox
          mutations made via the silent-action channel are visible on refresh
          (the snapshot in ``tool_calls.result`` is frozen at LLM-call time).

        Called by ``rich_media_parser.parse()`` exactly once per rich segment,
        immediately after the payload is extracted from the matching tool_calls
        row. Returning the same dict is fine; building a new one is fine too.

        Args:
            payload: The structured data extracted from the tool result string.
                ``dict`` for JSON-parsed bodies, kept as-is for raw strings.
            row: The matched ``tool_calls`` row dict (carries ``tool_name``,
                ``params``, ``result``, ``ephemeral``, ``created_at``).

        Returns:
            The payload to ship to the FE. Identity-returns the input by default.
        """
        return payload
