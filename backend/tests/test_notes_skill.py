import json
import pytest
from unittest.mock import patch
from services.memory_store import MemoryStore
from services.innate_skills.notes_skill import handle_notes

pytestmark = pytest.mark.unit


@pytest.fixture
def store():
    s = MemoryStore()
    with patch('services.memory_client.MemoryClientService.create_connection', return_value=s):
        yield s


def _write_entry(store, loop_id, entry):
    store.rpush(f"scratchpad:{loop_id}:entries", json.dumps(entry))


class TestNotesList:
    def test_empty_scratchpad(self, store):
        result = handle_notes("test", {"action": "list", "loop_id": "loop_123"})
        assert "empty" in result.lower()

    def test_list_entries(self, store):
        _write_entry(store, "loop_1", {"id": "sp_001", "source": "recall", "iteration": 1, "summary": "Found something important"})
        _write_entry(store, "loop_1", {"id": "sp_002", "source": "read", "iteration": 2, "summary": "Web page content"})
        result = handle_notes("test", {"action": "list", "loop_id": "loop_1"})
        assert "sp_001" in result
        assert "sp_002" in result
        assert "2 entries" in result


class TestNotesRead:
    def test_read_by_id(self, store):
        _write_entry(store, "loop_1", {
            "id": "sp_001", "source": "recall", "iteration": 1,
            "summary": "short", "full_content": "This is the full detailed content of the result",
        })
        result = handle_notes("test", {"action": "read", "id": "sp_001", "loop_id": "loop_1"})
        assert "full detailed content" in result

    def test_read_by_query(self, store):
        _write_entry(store, "loop_1", {
            "id": "sp_001", "source": "recall", "iteration": 1,
            "summary": "Python debugging tips", "full_content": "Use pdb for debugging", "query_hint": "debugging",
        })
        _write_entry(store, "loop_1", {
            "id": "sp_002", "source": "read", "iteration": 2,
            "summary": "JavaScript frameworks", "full_content": "React vs Vue comparison", "query_hint": "frameworks",
        })
        result = handle_notes("test", {"action": "read", "query": "debugging", "loop_id": "loop_1"})
        assert "pdb" in result
        assert "React" not in result

    def test_read_not_found(self, store):
        _write_entry(store, "loop_1", {"id": "sp_001", "source": "recall", "iteration": 1, "summary": "exists"})
        result = handle_notes("test", {"action": "read", "id": "sp_999", "loop_id": "loop_1"})
        assert "not found" in result.lower()

    def test_no_loop_id(self, store):
        result = handle_notes("test", {"action": "list", "loop_id": ""})
        assert "No active scratchpad" in result


class TestSizeGating:
    def test_small_result_stays_inline(self, store):
        from services.act_loop_service import ActLoopService
        loop = ActLoopService(config={}, loop_id="test_loop", scratchpad_enabled=True)
        result = {"action_type": "recall", "status": "ok", "result": "Short result", "execution_time": 0.1}
        loop.append_results([result])
        assert loop.act_history[0]["result"] == "Short result"
        assert "scratchpad_ref" not in loop.act_history[0]

    def test_large_result_truncated(self, store):
        from services.act_loop_service import ActLoopService
        loop = ActLoopService(config={}, loop_id="test_loop", scratchpad_enabled=True)
        large_text = "word " * 1200
        result = {"action_type": "read", "status": "ok", "result": large_text, "execution_time": 0.5}
        loop.append_results([result])
        assert "scratchpad_ref" in loop.act_history[0]
        assert "[full result in notes" in loop.act_history[0]["result"]
        entries = store.lrange("scratchpad:test_loop:entries", 0, -1)
        assert len(entries) == 1
        entry = json.loads(entries[0])
        assert "word" in entry["full_content"]
