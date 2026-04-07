"""
Unit tests for AutobiographyService.

Uses a real in-memory SQLite database with the actual schema so all SQL
queries execute against real tables.  No mocks of the DB layer.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager

import pytest

from services.autobiography_service import AutobiographyService
from services.database_service import DatabaseService


pytestmark = pytest.mark.unit


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """DatabaseService backed by a real SQLite file with the full schema."""
    from tests.test_helpers import load_schema_sql

    db_path = str(tmp_path / "test_autobiography.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(load_schema_sql())
    conn.commit()
    conn.close()

    svc = DatabaseService.__new__(DatabaseService)
    svc.db_path = db_path

    @contextmanager
    def _session():
        from services.database_service import SessionProxy
        c = sqlite3.connect(db_path)
        proxy = SessionProxy(c)
        try:
            yield proxy
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    svc.get_session = _session

    @contextmanager
    def _conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    svc.connection = _conn
    return svc


@pytest.fixture
def service(db_service):
    return AutobiographyService(db_service)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _insert_autobiography(db_service, version=1, narrative="Test narrative",
                          episode_cursor=None, episodes_since=5):
    """Insert an autobiography row directly."""
    aid = str(uuid.uuid4())
    with db_service.connection() as conn:
        conn.execute(
            """INSERT INTO autobiography
               (id, version, narrative, episode_cursor, episodes_since, section_hashes)
               VALUES (?, ?, ?, ?, ?, '{}')""",
            (aid, version, narrative, episode_cursor, episodes_since)
        )
    return aid


def _insert_episode(db_service, gist="test episode", salience=5, topic="general",
                    emotion="{}", outcome="ok", created_at=None):
    """Insert a minimal episode row."""
    eid = str(uuid.uuid4())
    with db_service.connection() as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, intent, context, action, emotion, outcome, gist,
                salience, topic, created_at)
               VALUES (?, '{}', '{}', 'act', ?, ?, ?, ?, ?, ?)""",
            (eid, emotion, outcome, gist, salience, topic,
             created_at or "2026-01-15T12:00:00")
        )
    return eid


def _insert_trait(db_service, key="fav_color", value="blue",
                  category="preference", confidence=0.8):
    """Insert a knowledge row of kind 'trait' (replaces old user_traits)."""
    with db_service.connection() as conn:
        conn.execute(
            """INSERT INTO knowledge (kind, entity, key, value, data, confidence)
               VALUES ('trait', 'user', ?, ?, ?, ?)""",
            (key, value, json.dumps({"category": category}), confidence)
        )
    # Return the rowid of the inserted row
    with db_service.connection() as conn:
        row = conn.execute(
            "SELECT id FROM knowledge WHERE key = ? ORDER BY id DESC LIMIT 1", (key,)
        ).fetchone()
    return str(row[0]) if row else None


def _insert_concept(db_service, name="Python", ctype="knowledge",
                    definition="A programming language", domain="tech",
                    strength=0.9):
    """Insert a knowledge row of kind 'concept' (replaces old semantic_concepts)."""
    with db_service.connection() as conn:
        conn.execute(
            """INSERT INTO knowledge (kind, entity, key, value, data, confidence)
               VALUES ('concept', 'chalie', ?, ?, ?, ?)""",
            (name, definition,
             json.dumps({"concept_type": ctype, "domain": domain, "strength": strength}),
             strength)
        )
    with db_service.connection() as conn:
        row = conn.execute(
            "SELECT id FROM knowledge WHERE key = ? AND kind = 'concept' ORDER BY id DESC LIMIT 1",
            (name,)
        ).fetchone()
    return str(row[0]) if row else None


def _insert_relationship(db_service, source_id, target_id, rel_type="uses",
                          strength=0.85):
    """Insert a knowledge row of kind 'relationship' (replaces old semantic_relationships)."""
    # Resolve source and target concept names from their IDs
    with db_service.connection() as conn:
        src_row = conn.execute("SELECT key FROM knowledge WHERE id = ?", (source_id,)).fetchone()
        tgt_row = conn.execute("SELECT key FROM knowledge WHERE id = ?", (target_id,)).fetchone()
    source_name = src_row[0] if src_row else "unknown"
    target_name = tgt_row[0] if tgt_row else "unknown"
    rel_key = f"{source_name}__{rel_type}__{target_name}"
    with db_service.connection() as conn:
        conn.execute(
            """INSERT INTO knowledge (kind, entity, key, value, data, confidence)
               VALUES ('relationship', 'chalie', ?, ?, ?, ?)""",
            (rel_key, f"{source_name} {rel_type} {target_name}",
             json.dumps({"source_name": source_name, "target_name": target_name,
                         "relationship_type": rel_type, "strength": strength}),
             strength)
        )
    with db_service.connection() as conn:
        row = conn.execute(
            "SELECT id FROM knowledge WHERE key = ? ORDER BY id DESC LIMIT 1", (rel_key,)
        ).fetchone()
    return str(row[0]) if row else None


# ─── get_current_narrative ────────────────────────────────────────────────────

class TestGetCurrentNarrative:

    def test_returns_none_when_empty(self, service):
        """Returns None when no autobiography rows exist."""
        assert service.get_current_narrative() is None

    def test_returns_latest_version(self, db_service, service):
        """Returns the highest-version row when multiple exist."""
        _insert_autobiography(db_service, version=1, narrative="First draft")
        _insert_autobiography(db_service, version=2, narrative="Second draft",
                              episodes_since=10)

        result = service.get_current_narrative()

        assert result is not None
        assert result["version"] == 2
        assert result["narrative"] == "Second draft"
        assert result["episodes_since"] == 10

    def test_maps_all_fields(self, db_service, service):
        """Row tuple is correctly mapped to dict with expected keys."""
        _insert_autobiography(db_service, version=1, narrative="My story",
                              episode_cursor="2026-01-15T12:00:00",
                              episodes_since=7)

        result = service.get_current_narrative()

        assert set(result.keys()) == {"id", "version", "narrative",
                                       "created_at", "episodes_since"}
        assert result["version"] == 1
        assert result["narrative"] == "My story"
        assert result["episodes_since"] == 7
        assert result["created_at"] is not None


# ─── should_synthesize ────────────────────────────────────────────────────────

class TestShouldSynthesize:

    def test_false_when_no_episodes(self, service):
        """No autobiography and no episodes -> False."""
        assert service.should_synthesize() is False

    def test_false_with_fewer_than_5_episodes(self, db_service, service):
        """No autobiography and <5 episodes -> False."""
        for i in range(4):
            _insert_episode(db_service, gist=f"ep {i}")

        assert service.should_synthesize() is False

    def test_true_with_5_episodes_no_autobiography(self, db_service, service):
        """No autobiography and >=5 episodes -> True."""
        for i in range(5):
            _insert_episode(db_service, gist=f"ep {i}")

        assert service.should_synthesize() is True

    def test_true_with_3_new_episodes_since_cursor(self, db_service, service):
        """Autobiography exists with cursor, 3+ new episodes after cursor -> True."""
        cursor_ts = "2026-01-10T00:00:00"
        _insert_autobiography(db_service, version=1, narrative="v1",
                              episode_cursor=cursor_ts)

        # Insert 3 episodes AFTER the cursor
        for i in range(3):
            _insert_episode(db_service, gist=f"new ep {i}",
                            created_at="2026-01-15T12:00:00")

        assert service.should_synthesize() is True

    def test_false_with_fewer_than_3_new_episodes(self, db_service, service):
        """Autobiography exists with cursor, <3 new episodes -> False."""
        cursor_ts = "2026-01-10T00:00:00"
        _insert_autobiography(db_service, version=1, narrative="v1",
                              episode_cursor=cursor_ts)

        # Insert 2 episodes AFTER cursor
        for i in range(2):
            _insert_episode(db_service, gist=f"new ep {i}",
                            created_at="2026-01-15T12:00:00")

        assert service.should_synthesize() is False


# ─── gather_synthesis_inputs ──────────────────────────────────────────────────

class TestGatherSynthesisInputs:

    def test_empty_db_returns_empty_lists(self, service):
        """All lists empty when no data exists."""
        result = service.gather_synthesis_inputs()

        assert result["episodes"] == []
        assert result["traits"] == []
        assert result["concepts"] == []
        assert result["relationships"] == []
        assert result["constraint_episodes"] == []

    def test_gathers_episodes_sorted_by_salience(self, db_service, service):
        """Episodes returned in descending salience order."""
        _insert_episode(db_service, gist="low salience", salience=2)
        _insert_episode(db_service, gist="high salience", salience=9)

        result = service.gather_synthesis_inputs()

        assert len(result["episodes"]) == 2
        assert result["episodes"][0]["gist"] == "high salience"
        assert result["episodes"][1]["gist"] == "low salience"

    def test_truncates_long_gists(self, db_service, service):
        """Gists over 500 chars are truncated with ellipsis."""
        long_gist = "x" * 600
        _insert_episode(db_service, gist=long_gist)

        result = service.gather_synthesis_inputs()

        gist = result["episodes"][0]["gist"]
        assert len(gist) == 503
        assert gist.endswith("...")

    def test_gathers_traits_above_confidence_threshold(self, db_service, service):
        """Only traits with confidence > 0.3 are included."""
        _insert_trait(db_service, key="visible", value="yes", confidence=0.5)
        _insert_trait(db_service, key="hidden", value="no", confidence=0.2)

        result = service.gather_synthesis_inputs()

        assert len(result["traits"]) == 1
        assert result["traits"][0]["key"] == "visible"
        assert result["traits"][0]["value"] == "yes"
        assert result["traits"][0]["category"] == "preference"

    def test_gathers_concepts(self, db_service, service):
        """Concepts are gathered with correct field mapping."""
        _insert_concept(db_service, name="Flask", ctype="framework",
                        definition="A web framework", domain="tech", strength=0.7)

        result = service.gather_synthesis_inputs()

        assert len(result["concepts"]) == 1
        assert result["concepts"][0] == {
            "name": "Flask",
            "type": "framework",
            "definition": "A web framework",
            "domain": "tech",
            "strength": 0.7,
        }

    def test_gathers_relationships(self, db_service, service):
        """Relationships resolve concept IDs to names via JOIN."""
        src_id = _insert_concept(db_service, name="Python", strength=0.9)
        tgt_id = _insert_concept(db_service, name="Flask", strength=0.7)
        _insert_relationship(db_service, src_id, tgt_id, "uses", 0.85)

        result = service.gather_synthesis_inputs()

        assert len(result["relationships"]) == 1
        assert result["relationships"][0] == {
            "source": "Python",
            "target": "Flask",
            "type": "uses",
            "strength": 0.85,
        }

    def test_gathers_constraint_episodes(self, db_service, service):
        """Episodes with outcome='constraint_learned' are gathered separately."""
        _insert_episode(db_service, gist="blocked by timing_gate",
                        outcome="constraint_learned")
        _insert_episode(db_service, gist="normal episode", outcome="ok")

        result = service.gather_synthesis_inputs()

        assert len(result["constraint_episodes"]) == 1
        assert "timing_gate" in result["constraint_episodes"][0]["gist"]

    def test_respects_since_cursor(self, db_service, service):
        """When since_cursor is provided, only episodes after it are returned."""
        _insert_episode(db_service, gist="old", created_at="2026-01-01T00:00:00")
        _insert_episode(db_service, gist="new", created_at="2026-02-01T00:00:00")

        result = service.gather_synthesis_inputs(
            since_cursor="2026-01-15T00:00:00"
        )

        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["gist"] == "new"

    def test_excludes_deleted_episodes(self, db_service, service):
        """Soft-deleted episodes are excluded."""
        eid = _insert_episode(db_service, gist="deleted one")
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE episodes SET deleted_at = '2026-01-20T00:00:00' WHERE id = ?",
                (eid,)
            )

        result = service.gather_synthesis_inputs()
        assert len(result["episodes"]) == 0

    def test_limits_episodes_to_50(self, db_service, service):
        """At most 50 episodes are returned."""
        for i in range(55):
            _insert_episode(db_service, gist=f"ep {i}", salience=5)

        result = service.gather_synthesis_inputs()
        assert len(result["episodes"]) == 50


# ─── _build_synthesis_prompt ──────────────────────────────────────────────────

class TestBuildSynthesisPrompt:

    def test_fresh_synthesis_includes_new_episodes_header(self, service):
        """First synthesis (no current) uses 'New Episodes' header."""
        inputs = {"episodes": [{"gist": "first day", "emotion": "excited",
                                 "topic": "onboarding"}],
                  "traits": [], "concepts": [], "relationships": [],
                  "constraint_episodes": []}

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "## New Episodes" in prompt
        assert "Current Narrative" not in prompt
        assert "first day" in prompt

    def test_incremental_synthesis_includes_current_narrative(self, service):
        """Incremental update includes current narrative and 'Since Last Synthesis' header."""
        inputs = {"episodes": [{"gist": "latest event", "emotion": "neutral",
                                 "topic": "work"}],
                  "traits": [], "concepts": [], "relationships": [],
                  "constraint_episodes": []}
        current = {"narrative": "Previously synthesized text", "version": 1}

        prompt = service._build_synthesis_prompt(inputs, current)

        assert "Current Narrative" in prompt
        assert "Previously synthesized text" in prompt
        assert "New Episodes Since Last Synthesis" in prompt

    def test_includes_traits_and_concepts(self, service):
        """Prompt includes formatted traits and concepts sections."""
        inputs = {
            "episodes": [{"gist": "ep", "emotion": "happy", "topic": "t"}],
            "traits": [{"key": "likes_coffee", "value": "true",
                        "confidence": 0.95, "category": "preference"}],
            "concepts": [{"name": "REST", "definition": "API style",
                          "strength": 0.8, "domain": "tech"}],
            "relationships": [],
            "constraint_episodes": [],
        }

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "## Observed Traits" in prompt
        assert "likes_coffee" in prompt
        assert "0.95" in prompt
        assert "## Key Concepts" in prompt
        assert "REST" in prompt

    def test_includes_constraint_episodes_when_present(self, service):
        """Constraint episodes appear under 'Learned Constraints' header."""
        inputs = {
            "episodes": [],
            "traits": [], "concepts": [], "relationships": [],
            "constraint_episodes": [
                {"gist": "communicate blocked by timing_gate",
                 "action": "learned constraint",
                 "created_at": "2026-03-07T12:00:00",
                 "retrieval_weight": 1.0}
            ],
        }

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "## Learned Constraints" in prompt
        assert "communicate blocked by timing_gate" in prompt

    def test_omits_empty_sections(self, service):
        """Sections with no data are omitted entirely."""
        inputs = {"episodes": [], "traits": [], "concepts": [],
                  "relationships": [], "constraint_episodes": []}

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "## Observed Traits" not in prompt
        assert "## Key Concepts" not in prompt
        assert "## Learned Constraints" not in prompt


# ─── _compute_section_hashes ─────────────────────────────────────────────────

class TestComputeSectionHashes:

    def test_produces_hashes_for_each_section(self, service):
        """Each ## section gets a truncated SHA-256 hash."""
        narrative = "## Identity\nI am Chalie.\n\n## Values\nI value attention."
        hashes = service._compute_section_hashes(narrative)

        assert "identity" in hashes
        assert "values" in hashes
        assert len(hashes["identity"]) == 16
        assert len(hashes["values"]) == 16

    def test_stable_for_same_content(self, service):
        """Same content produces identical hashes."""
        narrative = "## Section A\nContent A"
        h1 = service._compute_section_hashes(narrative)
        h2 = service._compute_section_hashes(narrative)
        assert h1 == h2

    def test_different_content_different_hashes(self, service):
        """Different content produces different hashes."""
        h1 = service._compute_section_hashes("## Section\nContent A")
        h2 = service._compute_section_hashes("## Section\nContent B")
        assert h1["section"] != h2["section"]


# ─── _store_narrative ─────────────────────────────────────────────────────────

class TestStoreNarrative:

    def test_inserts_first_version(self, db_service, service):
        """First call inserts version 1."""
        with db_service.get_session() as session:
            service._store_narrative(
                session, "## Identity\nI am Chalie.",
                "2026-01-15T12:00:00", 5, 1200
            )

        # Verify in DB
        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT version, narrative, episodes_since, synthesis_ms "
                "FROM autobiography ORDER BY version DESC LIMIT 1"
            ).fetchone()

        assert row["version"] == 1
        assert row["narrative"] == "## Identity\nI am Chalie."
        assert row["episodes_since"] == 5
        assert row["synthesis_ms"] == 1200

    def test_increments_version(self, db_service, service):
        """Second call auto-increments to version 2."""
        _insert_autobiography(db_service, version=1, narrative="v1")

        with db_service.get_session() as session:
            service._store_narrative(
                session, "v2 narrative",
                "2026-02-01T00:00:00", 3, 800
            )

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT version, narrative FROM autobiography "
                "ORDER BY version DESC LIMIT 1"
            ).fetchone()

        assert row["version"] == 2
        assert row["narrative"] == "v2 narrative"

    def test_stores_section_hashes_as_json(self, db_service, service):
        """Section hashes are stored as JSON in the section_hashes column."""
        with db_service.get_session() as session:
            service._store_narrative(
                session, "## Identity\nChalie\n\n## Values\nAttention",
                "2026-01-15T00:00:00", 5, 500
            )

        with db_service.connection() as conn:
            raw = conn.execute(
                "SELECT section_hashes FROM autobiography LIMIT 1"
            ).fetchone()["section_hashes"]

        hashes = json.loads(raw)
        assert "identity" in hashes
        assert "values" in hashes
