"""
Feature tests for ListAbility — strict CRUD interface.

Existing lists are addressed by ``id`` (8-char hex) or ``name`` for most
actions. ``name`` is used only for ``create`` and as the NEW name for
``rename``. The LLM discovers ids by calling ``list_all`` first, or can
pass the list name directly.

Actions: list_all, create, view, add, check, remove, clear, rename, delete

Tests use the real ``db`` fixture (per-test SQLite from schema.sql) — no
mocked services.
"""

import json

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture
def service(db):
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


@pytest.fixture
def ability(db):
    """ListAbility wired to the per-test SQLite via the ``db`` fixture."""
    from abilities.list import ListAbility
    return ListAbility()


def _exec(ability, params):
    """Run ListAbility.execute on the user channel without rich-media."""
    return ability.execute('user', params, telemetry=None)


def _exec_rich(ability, params, ordinal=1):
    """Run ListAbility.execute with a rich-media ordinal injected."""
    p = dict(params)
    p['_rich_media_ordinal'] = ordinal
    return ability.execute('user', p, telemetry=None)


def _payload(result):
    """Extract the JSON body from a non-rich result wrapped in skill tags.

    Format: ``[list(action=X)]\\n{json}\\n[end:list]``
    """
    text = result['text']
    start = text.index('\n') + 1
    end = text.rindex('\n[end:')
    return json.loads(text[start:end])


# ─── action: list_all ──────────────────────────────────────────────────────

class TestListAll:
    def test_returns_empty_when_no_lists(self, ability):
        body = _payload(_exec(ability, {"action": "list_all"}))
        assert body == {"status": "success", "lists": []}

    def test_returns_summaries_with_ids(self, ability, service):
        a = service.create_list("Groceries")
        b = service.create_list("Chores")
        service.add_items(a, ["milk", "eggs"])
        service.check_items(a, ["milk"])

        body = _payload(_exec(ability, {"action": "list_all"}))
        ids = {lst['id']: lst for lst in body['lists']}

        assert a in ids and b in ids
        assert ids[a]['name'] == 'Groceries'
        assert ids[a]['item_count'] == 2
        assert ids[a]['checked_count'] == 1
        assert ids[b]['item_count'] == 0


# ─── action: create ────────────────────────────────────────────────────────

class TestCreate:
    def test_creates_list_and_returns_payload_with_id(self, ability):
        body = _payload(_exec(ability, {"action": "create", "name": "Groceries"}))
        assert body['status'] == 'success'
        assert body['list']['name'] == 'Groceries'
        assert len(body['list']['id']) == 8
        assert body['list']['items'] == []

    def test_creates_with_initial_items(self, ability):
        body = _payload(_exec(ability, {
            "action": "create",
            "name": "Groceries",
            "items": ["milk", "eggs", "bread"],
        }))
        contents = [i['content'] for i in body['list']['items']]
        assert contents == ['milk', 'eggs', 'bread']

    def test_missing_name_fails(self, ability):
        body = _payload(_exec(ability, {"action": "create"}))
        assert body['status'] == 'fail'
        assert "'name'" in body['message']

    def test_duplicate_name_fails(self, ability, service):
        service.create_list("Groceries")
        body = _payload(_exec(ability, {"action": "create", "name": "Groceries"}))
        assert body['status'] == 'fail'
        assert 'already exists' in body['message']


# ─── action: view ──────────────────────────────────────────────────────────

class TestView:
    def test_returns_list_by_id(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])

        body = _payload(_exec(ability, {"action": "view", "id": list_id}))
        assert body['status'] == 'success'
        assert body['list']['id'] == list_id
        assert body['list']['items'][0]['content'] == 'milk'

    def test_missing_id_fails(self, ability):
        body = _payload(_exec(ability, {"action": "view"}))
        assert body['status'] == 'fail'
        assert "'id'" in body['message']

    def test_unknown_id_fails(self, ability):
        body = _payload(_exec(ability, {"action": "view", "id": "deadbeef"}))
        assert body['status'] == 'fail'
        assert 'not found' in body['message']

    def test_resolves_by_name(self, ability, service):
        list_id = service.create_list("Grocery List")
        service.add_items(list_id, ["milk"])
        body = _payload(_exec(ability, {"action": "view", "name": "Grocery List"}))
        assert body['status'] == 'success'
        assert body['list']['id'] == list_id

    def test_resolves_by_name_case_insensitive(self, ability, service):
        list_id = service.create_list("Grocery List")
        body = _payload(_exec(ability, {"action": "view", "name": "grocery list"}))
        assert body['status'] == 'success'
        assert body['list']['id'] == list_id

    def test_name_not_found_fails(self, ability):
        body = _payload(_exec(ability, {"action": "view", "name": "No Such List"}))
        assert body['status'] == 'fail'
        assert 'No list named' in body['message']


# ─── action: add ───────────────────────────────────────────────────────────

class TestAdd:
    def test_adds_items_by_id(self, ability, service):
        list_id = service.create_list("Groceries")
        body = _payload(_exec(ability, {
            "action": "add", "id": list_id, "items": ["milk", "eggs"],
        }))
        assert body['status'] == 'success'
        contents = [i['content'] for i in body['list']['items']]
        assert contents == ['milk', 'eggs']

    def test_missing_id_fails(self, ability):
        body = _payload(_exec(ability, {"action": "add", "items": ["milk"]}))
        assert body['status'] == 'fail'

    def test_missing_items_fails(self, ability, service):
        list_id = service.create_list("Groceries")
        body = _payload(_exec(ability, {"action": "add", "id": list_id}))
        assert body['status'] == 'fail'

    def test_unknown_id_fails(self, ability):
        body = _payload(_exec(ability, {
            "action": "add", "id": "deadbeef", "items": ["milk"],
        }))
        assert body['status'] == 'fail'
        assert 'not found' in body['message']

    def test_all_duplicates_fails(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])
        body = _payload(_exec(ability, {
            "action": "add", "id": list_id, "items": ["milk"],
        }))
        assert body['status'] == 'fail'
        assert 'duplicates' in body['message'].lower() or 'no items' in body['message'].lower()


# ─── action: check ─────────────────────────────────────────────────────────

class TestCheck:
    def test_checks_and_unchecks_in_one_call(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk", "eggs"])
        service.check_items(list_id, ["eggs"])

        body = _payload(_exec(ability, {
            "action": "check", "id": list_id,
            "items": [
                {"content": "milk", "checked": True},
                {"content": "eggs", "checked": False},
            ],
        }))
        assert body['status'] == 'success'
        by_content = {i['content']: i['checked'] for i in body['list']['items']}
        assert by_content == {'milk': True, 'eggs': False}

    def test_no_match_fails(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])
        body = _payload(_exec(ability, {
            "action": "check", "id": list_id,
            "items": [{"content": "bread", "checked": True}],
        }))
        assert body['status'] == 'fail'
        assert 'No matching items' in body['message']

    def test_malformed_items_fails(self, ability, service):
        list_id = service.create_list("Groceries")
        body = _payload(_exec(ability, {
            "action": "check", "id": list_id,
            "items": ["milk"],  # plain strings, not objects
        }))
        assert body['status'] == 'fail'
        assert "content" in body['message']

    def test_resolves_by_name(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])
        body = _payload(_exec(ability, {
            "action": "check", "name": "Groceries",
            "items": [{"content": "milk", "checked": True}],
        }))
        assert body['status'] == 'success'
        assert body['list']['items'][0]['checked'] is True


# ─── action: remove ────────────────────────────────────────────────────────

class TestRemove:
    def test_removes_items_by_id(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk", "eggs"])
        body = _payload(_exec(ability, {
            "action": "remove", "id": list_id, "items": ["milk"],
        }))
        assert body['status'] == 'success'
        assert [i['content'] for i in body['list']['items']] == ['eggs']

    def test_no_match_fails(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])
        body = _payload(_exec(ability, {
            "action": "remove", "id": list_id, "items": ["bread"],
        }))
        assert body['status'] == 'fail'
        assert 'No matching items' in body['message']


# ─── action: clear ─────────────────────────────────────────────────────────

class TestClear:
    def test_clears_all_items(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk", "eggs"])

        body = _payload(_exec(ability, {"action": "clear", "id": list_id}))
        assert body['status'] == 'success'
        assert body['list']['items'] == []

    def test_unknown_id_fails(self, ability):
        body = _payload(_exec(ability, {"action": "clear", "id": "deadbeef"}))
        assert body['status'] == 'fail'
        assert 'not found' in body['message']


# ─── action: rename ────────────────────────────────────────────────────────

class TestRename:
    def test_renames_by_id(self, ability, service):
        list_id = service.create_list("Old")
        body = _payload(_exec(ability, {
            "action": "rename", "id": list_id, "name": "New",
        }))
        assert body['status'] == 'success'
        assert body['list']['name'] == 'New'

    def test_collision_fails(self, ability, service):
        a = service.create_list("Old")
        service.create_list("Taken")
        body = _payload(_exec(ability, {
            "action": "rename", "id": a, "name": "Taken",
        }))
        assert body['status'] == 'fail'

    def test_missing_name_fails(self, ability, service):
        list_id = service.create_list("Old")
        body = _payload(_exec(ability, {"action": "rename", "id": list_id}))
        assert body['status'] == 'fail'

    def test_rename_requires_id_not_name(self, ability, service):
        """rename must require id — name is the new name, not a lookup key."""
        service.create_list("Old")
        body = _payload(_exec(ability, {"action": "rename", "name": "New"}))
        assert body['status'] == 'fail'
        assert "'id'" in body['message']


# ─── action: delete ────────────────────────────────────────────────────────

class TestDelete:
    def test_returns_exact_success_message(self, ability, service):
        list_id = service.create_list("Groceries")
        body = _payload(_exec(ability, {"action": "delete", "id": list_id}))
        assert body['status'] == 'success'
        assert body['message'] == f"List with id: {list_id} was deleted successfully"

    def test_unknown_id_fails(self, ability):
        body = _payload(_exec(ability, {"action": "delete", "id": "deadbeef"}))
        assert body['status'] == 'fail'
        assert 'not found' in body['message']

    def test_resolves_by_name(self, ability, service):
        list_id = service.create_list("Groceries")
        body = _payload(_exec(ability, {"action": "delete", "name": "Groceries"}))
        assert body['status'] == 'success'
        assert list_id in body['message']

    def test_missing_id_and_name_fails(self, ability):
        body = _payload(_exec(ability, {"action": "delete"}))
        assert body['status'] == 'fail'
        assert "'id' or 'name'" in body['message']


# ─── unknown action ────────────────────────────────────────────────────────

class TestUnknownAction:
    def test_unknown_action_fails_with_valid_list(self, ability):
        body = _payload(_exec(ability, {"action": "explode"}))
        assert body['status'] == 'fail'
        assert 'Unknown action' in body['message']
        assert 'list_all' in body['message']


# ─── rich-media envelope ───────────────────────────────────────────────────

def _unwrap_rich(result, action):
    """Extract (payload, instruction) from a rich-media result.

    Wire format: ``[list(action=X)]\\n{json}\\n\\n{instruction}\\n[end:list]``
    """
    assert isinstance(result, str)
    opener = f"[list(action={action})]\n"
    closer = "\n[end:list]"
    assert result.startswith(opener), f"missing opener: {result[:60]!r}"
    assert result.endswith(closer), f"missing closer: {result[-60:]!r}"
    inner = result[len(opener):-len(closer)]
    data_json, instruction = inner.split('\n\n', 1)
    return json.loads(data_json), instruction


class TestRichMedia:
    def test_create_emits_rich_payload_and_instruction(self, ability):
        result = _exec_rich(ability, {
            "action": "create", "name": "Groceries", "items": ["milk"],
        }, ordinal=1)
        payload, instruction = _unwrap_rich(result, "create")
        assert payload['name'] == 'Groceries'
        assert 'id' in payload
        assert "list_1" in instruction

    def test_add_emits_rich_payload(self, ability, service):
        list_id = service.create_list("Groceries")
        result = _exec_rich(ability, {
            "action": "add", "id": list_id, "items": ["milk"],
        }, ordinal=2)
        payload, instruction = _unwrap_rich(result, "add")
        assert payload['id'] == list_id
        assert "list_2" in instruction

    def test_check_emits_rich_payload(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])
        result = _exec_rich(ability, {
            "action": "check", "id": list_id,
            "items": [{"content": "milk", "checked": True}],
        }, ordinal=3)
        payload, _ = _unwrap_rich(result, "check")
        assert payload['items'][0]['checked'] is True

    def test_view_emits_rich_payload(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])
        result = _exec_rich(ability, {"action": "view", "id": list_id}, ordinal=4)
        payload, _ = _unwrap_rich(result, "view")
        assert payload['id'] == list_id

    def test_fail_falls_back_to_plain_skill_tag(self, ability):
        """Rich-media gate skips fail bodies — they go through the plain path
        with the `[list(action=X)]\\n{json}\\n[end:list]` wrapper and no
        instruction trailer."""
        result = _exec_rich(ability, {"action": "view", "id": "nonexist"}, ordinal=5)
        assert isinstance(result, dict)
        text = result['text']
        assert text.startswith("[list(action=view)]\n")
        assert text.endswith("\n[end:list]")
        assert "rich-media" not in text
        body = json.loads(text[len("[list(action=view)]\n"):-len("\n[end:list]")])
        assert body['status'] == 'fail'

    def test_delete_does_not_emit_rich_payload(self, ability, service):
        """Delete returns a message string, not a list. No rich card."""
        list_id = service.create_list("Groceries")
        result = _exec_rich(ability, {"action": "delete", "id": list_id}, ordinal=5)
        assert isinstance(result, dict)
        assert 'text' in result

    def test_rename_does_not_emit_rich_payload(self, ability, service):
        list_id = service.create_list("Old")
        result = _exec_rich(ability, {
            "action": "rename", "id": list_id, "name": "New",
        }, ordinal=6)
        assert isinstance(result, dict)
        assert 'text' in result

    def test_failure_does_not_emit_rich_payload(self, ability):
        """A failed call returns the standard text envelope, not a card."""
        result = _exec_rich(ability, {"action": "view", "id": "deadbeef"}, ordinal=7)
        assert isinstance(result, dict)
        assert 'text' in result


# ─── enrich_rich_payload (refresh hook) ────────────────────────────────────

class TestEnrichRichPayload:
    def test_returns_live_state_after_check(self, service):
        from abilities.list import ListAbility
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk", "eggs"])
        service.check_items(list_id, ["milk"])

        stale = {
            "id": list_id, "name": "Groceries",
            "items": [
                {"content": "milk", "checked": False},
                {"content": "eggs", "checked": False},
            ],
        }
        enriched = ListAbility.enrich_rich_payload(stale, row={})
        by_content = {i['content']: i['checked'] for i in enriched['items']}
        assert by_content == {"milk": True, "eggs": False}

    def test_falls_back_to_snapshot_when_list_deleted(self):
        from abilities.list import ListAbility
        snapshot = {
            "id": "deadbeef", "name": "Gone",
            "items": [{"content": "x", "checked": False}],
        }
        assert ListAbility.enrich_rich_payload(snapshot, row={}) == snapshot

    def test_passthrough_when_no_id(self):
        from abilities.list import ListAbility
        payload = {"name": "Anon", "items": []}
        assert ListAbility.enrich_rich_payload(payload, row={}) == payload
