"""
ListAbility — Manage deterministic lists via the ACT loop.

Lists and their items are addressed by NAME or id. ``add item="milk"
list="shopping"`` resolves the target by name; ``check items=["milk"]`` resolves
each item against the list's live contents. Resolution is consent-safe: an exact
id, an exact (case-insensitive) name, or a single substring match proceeds, but
when a partial name/term matches MORE than one candidate the call returns
``code=ambiguous-match`` listing every candidate and mutates nothing — it never
silently picks one. ``name`` is the NEW name for ``create`` and ``rename``.

Actions: list_all, create, view, add, check, remove, clear, rename, delete

Result contract: every action returns a :class:`abilities._result.ToolResult`
built only via ``ok()`` / ``err()``; the dispatcher renders the wire envelope.
List-returning actions emit rows ``{id, text, done, position}`` (``count`` =
item count); errors carry a stable kebab-case ``code`` (never the ``code="error"``
placeholder) so a weak model can self-correct without re-reading the schema.

Rich-media rendering:
  create/add/check/view return ``ToolResult.ok(body, rich=<fe_payload>)`` — the
  dispatcher (``ToolDispatcher._render``) owns ordinal assignment + the span-tag
  instruction and injects the card ONLY when the invoking channel broadcasts to
  the user. The rich payload keeps the LEGACY frontend shape (``{id, name,
  items:[{content, checked, id}]}``) the list card and ``enrich_rich_payload``
  depend on — distinct from the model-facing rows in the body, by design.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar, Optional, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.list_params_bag import (
    ListAddParams,
    ListAllParams,
    ListCheckParams,
    ListClearParams,
    ListCreateParams,
    ListDeleteParams,
    ListParamsBag,
    ListRemoveParams,
    ListRenameParams,
    ListViewParams,
)
from contracts.params.param_bag import ParamBag
from configs.enums.ability_category import AbilityCategory

if TYPE_CHECKING:
    from services.list_service import ListService as _ListService

logger = logging.getLogger(__name__)

_VALID_ACTIONS = (
    "list_all", "create", "view",
    "add", "check", "remove",
    "clear", "rename", "delete",
)


class ListAbility(Ability[ListParamsBag]):
    # Required params per action, validated by the dispatcher's ACTION_REQUIRED
    # pre-gate BEFORE the policy gate or run(). The target list is NOT pre-gated:
    # it is resolved ability-side (by name or the ``id`` alias) so the frontend
    # silent-action and old id-addressed transcripts survive the pre-gate. The
    # handlers raise the precise missing-target error.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "list_all": (),
        "create": (Keys.name_,),
        "view": (),
        "add": (Keys.items,),
        "check": (Keys.items,),
        "remove": (Keys.items,),
        "clear": (),
        "rename": (Keys.name_,),
        "delete": (),
    }

    # The typed input contract: the dispatch seam builds the bag via
    # ListParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = ListParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("checklist", "todo list", "shopping list", "lists")
    NAME: ClassVar[str] = "list"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.PRODUCTIVITY

    def get_summary(self) -> str:
        return "Create and manage named lists — shopping, to-do, chores — addressed by name or id."

    def get_examples(self) -> list[str]:
        return [
            "create a grocery list and add milk, eggs, and bread",
            "show me all my lists",
            "add bananas to the shopping list",
            "check off milk on the groceries list",
            "remove eggs from the shopping list",
            "rename my chores list to weekend chores",
            "delete list abc12345",
        ]

    def get_search_tooltip(self) -> str:
        return "list and checklist manager"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(_VALID_ACTIONS),
                "description": (
                    "list_all: return all lists. "
                    "create: make a new list. "
                    "view: read a list. "
                    "add: append items. "
                    "check: toggle checked state per item. "
                    "remove: remove specific items. "
                    "clear: empty a list but keep it. "
                    "rename: change a list's name. "
                    "delete: delete a list entirely."
                ),
            },
            Keys.list: {
                "type": "string",
                "description": (
                    "List name or id; addresses an existing list for "
                    "view/add/check/remove/clear/rename/delete. A partial name that "
                    "matches more than one list returns the candidates to pick from by id."
                ),
            },
            Keys.name_: {
                "type": "string",
                "description": "New list name; used for create and rename only.",
            },
            Keys.items: {
                "type": "array",
                "items": {},
                "description": (
                    "create/add/remove: string array, e.g. [\"milk\",\"eggs\"]. "
                    "check: object array, e.g. [{\"content\":\"milk\",\"checked\":true}]. "
                    "Item text may be a partial name; an ambiguous term returns the "
                    "matching items to pick from. Always a real JSON array, never a "
                    "serialised string. Optional for create."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: ListParamsBag) -> ToolResult:
        # The router factory only ever yields the nine leaves; a hand-built
        # foreign subclass (or the bare router) is rejected loudly BEFORE the
        # shared path validation reads a field it may not carry.
        if not isinstance(
            params,
            (
                ListAllParams,
                ListCreateParams,
                ListViewParams,
                ListAddParams,
                ListCheckParams,
                ListRemoveParams,
                ListClearParams,
                ListRenameParams,
                ListDeleteParams,
            ),
        ):
            return ToolResult.err(
                f"Unknown list params bag: {type(params).__name__}.",
                code="unknown-action",
                valid=_VALID_ACTIONS,
            )

        from services.list_service import ListService

        service = ListService()

        if isinstance(params, ListAllParams):
            return self._handle_list_all(service)
        if isinstance(params, ListCreateParams):
            return self._handle_create(service, params)
        if isinstance(params, ListViewParams):
            return self._handle_view(service, params)
        if isinstance(params, ListAddParams):
            return self._handle_add(service, params)
        if isinstance(params, ListCheckParams):
            return self._handle_check(service, params)
        if isinstance(params, ListRemoveParams):
            return self._handle_remove(service, params)
        if isinstance(params, ListClearParams):
            return self._handle_clear(service, params)
        if isinstance(params, ListRenameParams):
            return self._handle_rename(service, params)
        return self._handle_delete(service, params)

    @classmethod
    def enrich_rich_payload(cls, payload: dict[str, object]) -> dict[str, object]:
        """Re-fetch the live list so checkbox state the user toggled out-of-band
        is visible on conversation refresh.

        ``tool_calls.result`` is a frozen snapshot from the LLM-call moment; the
        FE list card writes each toggle directly to the ``lists`` table via
        ``POST /api/lists/items/<list_id>:<item_id>`` (``ListService.update_item``).
        Without this hook, refresh would replay the stale snapshot and visually
        un-tick boxes the user already ticked. Single read path: both the live
        handler and the refresh path go through ``ListService.get_list``
        (operating on the legacy FE payload shape).
        """
        list_id = payload.get("id") if isinstance(payload, dict) else None
        if not list_id:
            return payload
        from services.list_service import ListService

        service = ListService()
        fresh = _fe_payload(service, cast(str, list_id))
        return fresh if fresh else payload

    def _handle_list_all(self, service: "_ListService") -> ToolResult:
        rows: list[dict[str, object]] = [
            {
                "id": lst["id"],
                "name": lst["name"],
                "item_count": lst["item_count"],
                "checked_count": lst["checked_count"],
            }
            for lst in service.get_all_lists()
        ]
        if not rows:
            return ToolResult.no_results()
        return ToolResult.ok(rows, count=len(rows))

    def _handle_create(self, service: "_ListService", params: ListCreateParams) -> ToolResult:
        name = params.name_
        items = _normalize_string_items(params.items)
        try:
            list_id = service.create_list(name)
        except ValueError as e:
            return ToolResult.err(str(e), code="duplicate-name", hint="choose a different name.")

        if items:
            added = service.add_items(list_id, items)
            if added < len(items):
                logger.info(
                    "[LIST SKILL] create: added %d/%d items to list '%s' (id=%s) — some skipped (dedupe or empty)",
                    added, len(items), name, list_id,
                )
        return _ok_list(service, list_id)

    def _handle_view(self, service: "_ListService", params: ListViewParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst
        return _ok_list(service, cast(str, lst["id"]))

    def _handle_add(self, service: "_ListService", params: ListAddParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst

        cleaned = _normalize_string_items(params.items)
        if not cleaned:
            return ToolResult.err(
                "'items' must contain at least one non-empty string.",
                code="invalid-items",
                hint="pass a JSON array of item strings, e.g. [\"milk\",\"eggs\"].",
            )

        added = service.add_items(cast(str, lst["id"]), cleaned)
        if added == 0:
            return ToolResult.err(
                "No items added (all duplicates or empty).",
                code="no-items-added",
                hint="the items are already on the list.",
            )
        return _ok_list(service, cast(str, lst["id"]))

    def _handle_check(self, service: "_ListService", params: ListCheckParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst

        to_check, to_uncheck = _split_check_items(params.items)
        if not to_check and not to_uncheck:
            return ToolResult.err(
                "Each item must have 'content' (string) and 'checked' (bool).",
                code="invalid-items",
                hint="e.g. [{\"content\":\"milk\",\"checked\":true}].",
            )

        resolved_check = _resolve_items(lst, to_check)
        if isinstance(resolved_check, ToolResult):
            return resolved_check
        resolved_uncheck = _resolve_items(lst, to_uncheck)
        if isinstance(resolved_uncheck, ToolResult):
            return resolved_uncheck

        if resolved_check:
            service.check_items(cast(str, lst["id"]), resolved_check)
        if resolved_uncheck:
            service.uncheck_items(cast(str, lst["id"]), resolved_uncheck)
        return _ok_list(service, cast(str, lst["id"]))

    def _handle_remove(self, service: "_ListService", params: ListRemoveParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst

        cleaned = _normalize_string_items(params.items)
        if not cleaned:
            return ToolResult.err(
                "'items' must contain at least one non-empty string.",
                code="invalid-items",
                hint="pass a JSON array of item strings to remove.",
            )

        resolved = _resolve_items(lst, cleaned)
        if isinstance(resolved, ToolResult):
            return resolved

        service.remove_items(cast(str, lst["id"]), resolved)
        return _ok_list(service, cast(str, lst["id"]))

    def _handle_clear(self, service: "_ListService", params: ListClearParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst
        service.clear_list(cast(str, lst["id"]))
        return _ok_list(service, cast(str, lst["id"]))

    def _handle_rename(self, service: "_ListService", params: ListRenameParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst

        new_name = params.name_
        if not service.rename_list(cast(str, lst["id"]), new_name):
            return ToolResult.err(
                f"Rename failed — name {new_name!r} is already in use.",
                code="duplicate-name",
                hint="choose a different name.",
            )
        return _ok_list(service, cast(str, lst["id"]))

    def _handle_delete(self, service: "_ListService", params: ListDeleteParams) -> ToolResult:
        lst = _resolve_list(service, params)
        if isinstance(lst, ToolResult):
            return lst

        if not service.delete_list(cast(str, lst["id"])):
            return ToolResult.err(
                f"Failed to delete {lst['name']!r}.",
                code="delete-failed",
            )
        return ToolResult.ok({"id": lst["id"], "name": lst["name"], "deleted": True})


# ── Target resolution (by name or id) ───────────────────────────────────────────


def _resolve_list(service: "_ListService", params: "ListViewParams | ListAddParams | ListCheckParams | ListRemoveParams | ListClearParams | ListRenameParams | ListDeleteParams") -> "dict[str, object] | ToolResult":
    """Resolve the addressed list for view/add/check/remove/clear/rename/delete.

    Returns the service list row dict on an unambiguous resolution, or an error
    ``ToolResult`` (``missing-target`` / ``list-not-found`` / ``ambiguous-match``).
    Resolution order: exact id → exact (case-insensitive) name → single
    case-insensitive substring match. More than one substring match returns
    ``ambiguous-match`` with the candidate rows rather than silently picking one.
    The ``id`` spelling keeps the frontend silent-action and old transcripts
    working — it is healed to the canonical ``list`` key upstream at the dispatch
    seam via ``VARIANTS[Keys.list]``.
    """
    target = params.list_
    if not target:
        return ToolResult.err(
            "No list specified.",
            code="missing-target",
            hint="pass the list name or id as 'list'.",
        )

    # Exact id wins (8-char hex addressing path, incl. the FE silent-action).
    by_id = service.get_list(target)
    if by_id is not None:
        return by_id

    lists = service.get_all_lists()
    lowered = target.lower()
    exact = [lst for lst in lists if cast(str, lst["name"]).lower() == lowered]
    if len(exact) == 1:
        return cast("dict[str, object]", service.get_list(cast(str, exact[0]["id"])))

    substring = [lst for lst in lists if lowered in cast(str, lst["name"]).lower()]
    if len(substring) == 1:
        return cast("dict[str, object]", service.get_list(cast(str, substring[0]["id"])))
    if len(substring) > 1:
        return ToolResult.err(
            f"{target!r} matches {len(substring)} list(s); re-call with the exact id. "
            f"Candidates: {_candidate_json(substring)}",
            code="ambiguous-match",
            hint="re-issue the action with 'list' set to the chosen candidate id.",
            count=len(substring),
        )
    return ToolResult.err(
        f"No list matching {target!r} was found.",
        code="list-not-found",
        hint="use action=list_all to see available lists.",
    )


def _candidate_json(lists: list[dict[str, object]]) -> str:
    rows = [{"id": lst["id"], "name": lst["name"]} for lst in lists]
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


# ── Item resolution (check / remove) ────────────────────────────────────────────


def _resolve_items(lst: dict[str, object], terms: list[str]) -> "list[str] | ToolResult":
    """Resolve each requested item *term* against the list's active contents.

    For each term: exact (case-insensitive) match → kept as-is; else a single
    case-insensitive substring match → substituted with the full content; more
    than one substring match → ``ambiguous-match`` (nothing is mutated, the whole
    call aborts); zero → ``item-not-found``. Returns the resolved exact contents
    in request order, or an error ``ToolResult``.
    """
    contents = [cast(str, item["content"]) for item in cast(list[dict[str, object]], lst.get("items", []))]
    resolved: list[str] = []
    for term in terms:
        lowered = term.strip().lower()
        exact = [c for c in contents if c.lower() == lowered]
        if exact:
            resolved.append(exact[0])
            continue
        substring = [c for c in contents if lowered in c.lower()]
        if len(substring) == 1:
            resolved.append(substring[0])
            continue
        if len(substring) > 1:
            rows = json.dumps(substring, ensure_ascii=False, separators=(",", ":"))
            return ToolResult.err(
                f"{term!r} matches {len(substring)} item(s) in {lst['name']!r}; "
                f"re-call with the exact text. Candidates: {rows}",
                code="ambiguous-match",
                hint="re-issue the action with the exact item text.",
                count=len(substring),
            )
        return ToolResult.err(
            f"No item matching {term!r} in {lst['name']!r}.",
            code="item-not-found",
            hint="use action=view to see the list's items.",
        )
    return resolved


# ── Body / rich shaping ─────────────────────────────────────────────────────────


def _rows(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": item["id"],
            "text": item["content"],
            "done": bool(item["checked"]),
            "position": item["position"],
        }
        for item in items
    ]


def _body(lst: dict[str, object]) -> dict[str, object]:
    return {"id": lst["id"], "name": lst["name"], "items": _rows(cast("list[dict[str, object]]", lst.get("items", [])))}


def _fe_card(lst: dict[str, object]) -> dict[str, object]:
    return {
        "id": lst["id"],
        "name": lst["name"],
        "items": [
            {"content": item["content"], "checked": bool(item["checked"]), "id": item["id"]}
            for item in cast(list[dict[str, object]], lst.get("items", []))
        ],
    }


def _fe_payload(service: "_ListService", list_id: str) -> Optional[dict[str, object]]:
    """Re-fetch a list and shape it as the FE card payload (refresh-hook path)."""
    lst = service.get_list(list_id)
    return _fe_card(lst) if lst else None


def _ok_list(service: "_ListService", list_id: str) -> ToolResult:
    """Fetch the live list and return a success result with rows + the FE card."""
    lst = service.get_list(list_id)
    if lst is None:
        return ToolResult.err(
            f"No list with id {list_id!r}.",
            code="list-not-found",
            hint="use action=list_all to see available lists.",
        )
    return ToolResult.ok(_body(lst), rich=_fe_card(lst), count=len(cast(list[object], lst.get("items", []))))


def _normalize_string_items(items: list[object]) -> list[str]:
    """Item text from either shape the contract permits: a bare string, or the
    ``{"content": ...}`` object that ``check`` already requires and the REST item
    endpoint already speaks. ``items`` is declared untyped in the schema, so
    taking strings alone rejected schema-valid calls as 'must contain at least
    one non-empty string'."""
    texts = (i.get("content") if isinstance(i, dict) else i for i in items)
    return [t.strip() for t in texts if isinstance(t, str) and t.strip()]


def _split_check_items(raw_items: list[object]) -> tuple[list[str], list[str]]:
    to_check: list[str]
    to_uncheck: list[str]
    to_check, to_uncheck = [], []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if "checked" not in item:
            continue
        (to_check if item["checked"] else to_uncheck).append(content)
    return to_check, to_uncheck
