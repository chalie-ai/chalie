"""policy_defaults.json is a valid static seed (flat triples + internal rows)."""
import json

from services.file_mapper_service import FileMapperService

_CHANNELS = {"chat", "subagent", "subconscious", "external_agent"}
_SETTINGS = {"internal", "allow", "ask", "deny"}


def test_seed_is_valid_and_carries_known_rows():
    with open(FileMapperService.get_policy_defaults_path()) as f:
        rows = json.load(f)

    assert isinstance(rows, list) and rows
    keys = []
    for r in rows:
        assert set(r.keys()) == {"channel", "permission", "setting"}
        assert r["channel"] in _CHANNELS and r["setting"] in _SETTINGS
        assert isinstance(r["permission"], str) and r["permission"]
        keys.append((r["channel"], r["permission"]))
    assert len(keys) == len(set(keys)), "duplicate (channel, permission) in seed"

    by_key = {(r["channel"], r["permission"]): r["setting"] for r in rows}
    # visible defaults ported from the old matrix
    assert by_key[("chat", "email.search")] == "allow"
    assert by_key[("chat", "email.manage")] == "ask"
    assert by_key[("subconscious", "email.manage")] == "deny"
    # SYSTEM (sub)actions seeded internal in every channel (granular + bare)
    for ch in _CHANNELS:
        assert by_key[(ch, "memory.recall")] == "internal"
        assert by_key[(ch, "memory.forget")] == "internal"
        assert by_key[(ch, "timer")] == "internal"
