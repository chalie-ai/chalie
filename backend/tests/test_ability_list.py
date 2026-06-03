"""
Feature tests for ListAbility — strict CRUD interface.

Existing lists are addressed by ``id`` (8-char hex). ``name`` is used only
for ``create`` and ``rename``. The LLM discovers ids by calling
``list_all`` first.

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
    return ability.run(params)


def _exec_rich(ability, params, ordinal=1):
    """Run ListAbility.execute with a rich-media ordinal injected."""
    p = dict(params)
    p['_rich_media_ordinal'] = ordinal
    return ability.run(p)


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



# ─── action: view ──────────────────────────────────────────────────────────

class TestView:
    def test_returns_list_by_id(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk"])

        body = _payload(_exec(ability, {"action": "view", "id": list_id}))
        assert body['status'] == 'success'
        assert body['list']['id'] == list_id
        assert body['list']['items'][0]['content'] == 'milk'

    def test_unknown_id_fails(self, ability):
        body = _payload(_exec(ability, {"action": "view", "id": "deadbeef"}))
        assert body['status'] == 'fail'
        assert 'not found' in body['message']


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

    def test_unknown_id_fails(self, ability):
        body = _payload(_exec(ability, {
            "action": "add", "id": "deadbeef", "items": ["milk"],
        }))
        assert body['status'] == 'fail'
        assert 'not found' in body['message']




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




# ─── action: clear ─────────────────────────────────────────────────────────

class TestClear:
    def test_clears_all_items(self, ability, service):
        list_id = service.create_list("Groceries")
        service.add_items(list_id, ["milk", "eggs"])

        body = _payload(_exec(ability, {"action": "clear", "id": list_id}))
        assert body['status'] == 'success'
        assert body['list']['items'] == []


# ─── action: rename ────────────────────────────────────────────────────────

class TestRename:
    def test_renames_by_id(self, ability, service):
        list_id = service.create_list("Old")
        body = _payload(_exec(ability, {
            "action": "rename", "id": list_id, "name": "New",
        }))
        assert body['status'] == 'success'
        assert body['list']['name'] == 'New'


# ─── action: delete ────────────────────────────────────────────────────────

class TestDelete:
    def test_returns_exact_success_message(self, ability, service):
        list_id = service.create_list("Groceries")
        body = _payload(_exec(ability, {"action": "delete", "id": list_id}))
        assert body['status'] == 'success'
        assert body['message'] == f"List with id: {list_id} was deleted successfully"



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



