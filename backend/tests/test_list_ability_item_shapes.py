# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The ``items`` parameter accepts both shapes the tool contract declares.

``items`` is schema-typed as an untyped array (``{"type": "array", "items": {}}``)
and its description gives create/add/remove a string array but ``check`` an
object array. Taking only strings therefore rejected schema-valid input: a model
that sent the ``{"content": ...}`` object form — the same form ``check`` demands
and the REST item endpoint speaks — got ``'items' must contain at least one
non-empty string`` on every attempt and looped until the runaway guard fired.

These drive the real ability against the real database, so they prove the rows
actually land rather than that a helper returned a list.
"""

from typing import cast

import pytest

from abilities.list import ListAbility
from contracts.params.list_params_bag import ListParamsBag
from services.list_service import ListService
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit


def _run(params: dict[str, object]) -> object:
    return ListAbility().run(built(ListParamsBag.from_params(params)))


@pytest.fixture
def list_id(db: object) -> str:
    """The ``db`` fixture binds the real test database the ability writes through."""
    return ListService().create_list("Shopping List")


def _contents(list_id: str) -> list[str]:
    return [cast(str, i["content"]) for i in (ListService().get_items(list_id) or [])]


class TestAddAcceptsBothItemShapes:
    def test_object_form_lands_rows(self, list_id: str) -> None:
        result = _run({
            "action": "add",
            "list": "Shopping List",
            "items": [{"content": "olives"}, {"content": "capers"}],
        })
        assert getattr(result, "status", None) != "error", getattr(result, "body", result)
        assert _contents(list_id) == ["olives", "capers"]

    def test_object_form_with_checked_flag_lands_rows(self, list_id: str) -> None:
        """The exact payload the model sent when it looped — ``checked`` rides
        along on an ``add`` and must not disqualify the item."""
        _run({
            "action": "add",
            "list": "Shopping List",
            "items": [{"content": "olives", "checked": False}],
        })
        assert _contents(list_id) == ["olives"]

    def test_string_form_still_lands_rows(self, list_id: str) -> None:
        _run({"action": "add", "list": "Shopping List", "items": ["olives", "capers"]})
        assert _contents(list_id) == ["olives", "capers"]

    def test_shapes_are_interchangeable_within_one_call(self, list_id: str) -> None:
        _run({
            "action": "add",
            "list": "Shopping List",
            "items": ["olives", {"content": "capers"}],
        })
        assert _contents(list_id) == ["olives", "capers"]


class TestItemsWithNoTextStillRejected:
    """Widening the accepted shapes must not swallow genuinely empty input —
    the ``invalid-items`` error still has to fire, or a no-op add reports success."""

    @pytest.mark.parametrize("items", [
        [{"checked": True}],
        [{"content": "   "}],
        [{"content": None}],
        [42],
    ])
    def test_add_rejects_items_carrying_no_text(self, list_id: str, items: list[object]) -> None:
        result = _run({"action": "add", "list": "Shopping List", "items": items})
        assert getattr(result, "code", None) == "invalid-items"
        assert _contents(list_id) == []


class TestCreateAndRemoveAcceptObjectForm:
    def test_create_seeds_items_from_object_form(self, db: object) -> None:
        _run({
            "action": "create",
            "name": "Groceries",
            "items": [{"content": "olives"}, {"content": "capers"}],
        })
        created = next(lst for lst in ListService().get_all_lists() if lst["name"] == "Groceries")
        assert _contents(cast(str, created["id"])) == ["olives", "capers"]

    def test_remove_matches_object_form(self, list_id: str) -> None:
        _run({"action": "add", "list": "Shopping List", "items": ["olives", "capers"]})
        _run({"action": "remove", "list": "Shopping List", "items": [{"content": "olives"}]})
        assert _contents(list_id) == ["capers"]
