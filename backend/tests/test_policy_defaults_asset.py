"""policy_defaults.json is a valid static seed (flat triples + internal rows)."""
import json

import pytest

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_CHANNELS = {"chat", "subconscious", "external_agent"}
_SETTINGS = {"internal", "allow", "ask", "deny"}


def test_seed_is_valid_and_carries_known_rows() -> None:
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
    assert by_key[("chat", "bash.read")] == "allow"
    assert by_key[("chat", "bash.modify_file")] == "ask"
    # delegate tools gate at the OUTER permission only (inner surface is INTERNAL);
    # pim defaults to ask on chat and deny on every no-human channel
    assert by_key[("chat", "pim")] == "ask"
    assert by_key[("external_agent", "pim")] == "deny"
    assert by_key[("subconscious", "pim")] == "deny"
    # Infrastructure actions still seeded internal in every channel
    for ch in _CHANNELS:
        assert by_key[(ch, "timer")] == "internal"

    # PolicyManager.INTERNAL tools ALWAYS bypass the gate, so they carry NO seed
    # rows (the frozenset is the source of truth, not the JSON). See policy_manager.
    from services.policy_manager import INTERNAL
    tools_in_seed = {r["permission"].split(".", 1)[0] for r in rows}
    assert tools_in_seed.isdisjoint(INTERNAL), (
        f"INTERNAL tools must not appear in the seed: {tools_in_seed & INTERNAL}"
    )
