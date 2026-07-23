"""
ContactsAbility — Search and look up contacts from the local people index.

Both actions are READS answered inline against the data graph (the same
``user_specific`` rows the CardDAV ingest writes via
``contact_resolver.index_contact_profile``), so contacts work offline regardless
of whether the mail capability is currently connected — local data has no live
dependency. The capability/connected gate is consulted only as a fallback: when a
read finds NOTHING and mail is not connected, the base's ``code=not-connected``
remediation is surfaced so the user still learns how to populate the index.

The mail capability (CAPABILITY_KEY="mail") is consulted only inside
``_not_connected`` — this ability owns its whole ``run()`` flow and does not use
the ``CapabilityAbility`` handler-dispatch base.

``get`` carries a precision contract — ``resolve()`` is fuzzy (FTS5 prefix-match
over the contact kind, no score threshold), so taking ``matches[0]`` could hand
back the wrong person. A relevance predicate splits the fuzzy candidates into
three fixed outcomes (calendar / document precedent):

* exactly 1 relevant → that contact.
* ≥2 relevant → ``code=ambiguous-match`` with the candidate rows in the body —
  never a silent first-hit pick.
* 0 relevant → ``code=not-found`` with a closest-match hint naming the top fuzzy
  candidates (or the base not-connected error when the index is empty).

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``; the
rich contacts card travels via ``ToolResult(rich=…)`` and the dispatcher owns the
ordinal + the single span instruction. This ability never formats a wire
envelope.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from contracts.params.param_bag import ParamBag

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.contacts_params_bag import (
    ContactsGetParams,
    ContactsListParams,
    ContactsParamsBag,
)
from models.contact import ContactRow


class ContactsAbility(Ability[ContactsParamsBag]):
    DISCOVERABLE: ClassVar[bool] = False  # pim-delegate-exclusive; pinned on PimConfig only
    CAPABILITY_KEY: ClassVar[str] = "mail"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the mail integration in the Brain dashboard."
    )

    # The typed input contract: the dispatch seam builds the bag via
    # ContactsParamsBag.from_params before run() is called.
    PARAMS: ClassVar["type[ParamBag] | None"] = ContactsParamsBag

    # Pre-gated by the dispatcher BEFORE run(): get requires an identifier; list
    # requires nothing (an empty query lists the whole book). An unknown action
    # is rejected here and again by the bag's router.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "list": (),
        "get": (Keys.identifier,),
    }

    def get_name(self) -> str:
        return "contacts"

    def get_summary(self) -> str:
        return (
            "Search and look up contacts from the connected address book (CardDAV). "
            "Available when the user asks for someone's phone number, email, or contact details."
        )

    def get_examples(self) -> list[str]:
        return [
            "find John's phone number",
            "look up Sarah's email address",
            "who is in my contacts",
            "get me the contact details for Mike",
            "search my address book for someone named Alex",
            "what's the phone number for the dentist",
            "show me all my contacts",
        ]

    def get_search_tooltip(self) -> str:
        return "contact book"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["list", "get"],
                "description": (
                    "The contacts action to perform. "
                    "list — search contacts by name or keyword, or list all contacts. "
                    "get — fetch full details for a specific contact by name or email."
                ),
            },
            Keys.query: {
                "type": "string",
                "description": "list: search query to filter contacts by name or other fields.",
            },
            Keys.limit: {
                "type": "integer",
                "description": "list: maximum number of contacts to return.",
            },
            Keys.identifier: {
                "type": "string",
                "description": "get: name or email address identifying the contact to look up.",
            },
        },
        "required": [Keys.action],
    }

    # ── Entry point — both actions are inline reads against the local index ─────

    def run(self, params: ContactsParamsBag) -> ToolResult:
        # The router factory only ever yields the two leaves; the isinstance
        # chain is the narrowing that hands each handler its exact leaf type.
        # The trailing branch catches only a hand-built foreign subclass (or
        # the bare router) — loudly.
        if isinstance(params, ContactsListParams):
            return self._list(params)
        if isinstance(params, ContactsGetParams):
            return self._get(params)
        return ToolResult.err(
            f"Unknown contacts params bag: {type(params).__name__}.",
            code="unknown-action",
            valid=("list", "get"),
        )

    # ── list ────────────────────────────────────────────────────────────────

    def _list(self, params: ContactsListParams) -> ToolResult:
        from capabilities.contact_resolver import _parse_contact_row, resolve

        limit = params.limit
        query = params.query

        if query:
            contacts = resolve(query, limit=limit)
        else:
            rows = [
                r.to_dict()
                for r in ContactRow.live().order_by("key ASC").limit(limit).get()
            ]
            contacts = [
                c
                for c in (
                    _parse_contact_row(cast(str, r.get("key", "")), cast(str, r.get("value", "")))
                    for r in rows
                )
                if c is not None
            ]

        if not contacts:
            not_connected = self._not_connected("list")
            if not_connected is not None:
                return not_connected

        body = {
            "action_performed": "list",
            "contacts": contacts,
            "count": len(contacts),
        }
        return ToolResult.ok(
            body,
            rich=dict(body) if contacts else None,
            action="list",
            count=len(contacts),
        )

    # ── get — three-outcome precision contract ────────────────────────────────

    def _get(self, params: ContactsGetParams) -> ToolResult:
        from capabilities.contact_resolver import resolve

        identifier = params.identifier
        candidates = resolve(identifier, limit=5)

        if not candidates:
            not_connected = self._not_connected("get")
            if not_connected is not None:
                return not_connected
            return ToolResult.err(
                f"No contact matching {identifier!r} was found.",
                code="not-found",
                hint="call contacts with action=list to see what exists.",
                action="get",
            )

        relevant = [c for c in candidates if _is_relevant(identifier, c)]

        if len(relevant) == 1:
            body = {"action_performed": "get", "contact": relevant[0]}
            return ToolResult.ok(body, rich=dict(body), action="get")

        if len(relevant) >= 2:
            rows = [{"id": _contact_id(c), "name": _contact_name(c)} for c in relevant]
            return ToolResult.err(
                f"Multiple contacts match {identifier!r}: {json.dumps(rows, ensure_ascii=False)}.",
                code="ambiguous-match",
                hint="re-issue with the exact full name or email.",
                action="get",
            )

        # 0 relevant but fuzzy candidates exist — surface them as a closest-match
        # hint so the model can re-issue with an exact name, never a wrong pick.
        names = ", ".join(_contact_name(c) for c in candidates[:3])
        return ToolResult.err(
            f"No exact contact match for {identifier!r}.",
            code="not-found",
            hint=f"closest matches: {names} — re-issue with the exact name or email.",
            action="get",
        )

    # ── Not-connected fallback (only when a read found nothing) ────────────────

    def _not_connected(self, action: str) -> "ToolResult | None":
        """Return the base's not-connected error when mail is absent / not
        connected, else ``None`` (an empty index on a connected account is a real
        empty result, not a remediation case)."""
        from capabilities import load_capabilities

        cap = load_capabilities().get(self.CAPABILITY_KEY)
        if cap is None or not cap.is_connected():
            return ToolResult.err(
                f"{self.get_name().capitalize()} capability not connected.",
                code="not-connected",
                hint=self.NOT_CONNECTED_HINT,
                action=action,
            )
        return None


# ---------------------------------------------------------------------------
# Relevance predicate + contact field accessors
# ---------------------------------------------------------------------------


def _is_relevant(identifier: str, contact: dict[str, object]) -> bool:
    """A fuzzy candidate is *relevant* to *identifier* when the identifier
    ci-equals the contact's fn / name / nickname or any email value, OR when every
    word of the identifier appears as a ci word-prefix in the contact's fn (so
    "Mike" matches "Mike Borg" but "Smith Family" does not match "John Smith").

    This is the precision filter that turns the unthresholded FTS candidate set
    into the three fixed get outcomes — without it ``resolve()`` would hand back a
    near-arbitrary first hit for almost any query on a non-empty index."""
    needle = identifier.strip().lower()
    if not needle:
        return False

    name = _contact_name(contact).strip().lower()
    nickname = (cast(str, contact.get("nickname")) or "").strip().lower()
    if needle in {name, nickname}:
        return True

    for email in cast("list[dict[str, object]]", contact.get("emails") or []):
        value = (cast(str, email.get("value")) or "").strip().lower()
        if value and needle == value:
            return True
    legacy_email = (cast(str, contact.get("email")) or "").strip().lower()
    if legacy_email and needle == legacy_email:
        return True

    if name:
        name_words = name.split()
        return all(
            any(word.startswith(token) for word in name_words)
            for token in needle.split()
        )
    return False


def _contact_name(contact: dict[str, object]) -> str:
    return (cast(str, contact.get("fn")) or cast(str, contact.get("name")) or "").strip()


def _contact_id(contact: dict[str, object]) -> str:
    return (
        cast(str, contact.get("uid"))
        or _contact_name(contact)
        or cast(str, contact.get("email") or "")
    )
