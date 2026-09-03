# Baseline: 2249 passed, 65 failed, 499 errors (2026-03-27)
# Errors are pre-existing: 15 files excluded (numpy import failure in this env),
# and 499 test-setup errors caused by missing sqlite-vec extension (vec0 module).
import math
import re
import shutil
import sqlite3
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from services.memory_store import MemoryStore


# ── Real-SQLite fixtures ──────────────────────────────────────────
# Session-scoped template: full schema + migrations applied once.
# Function-scoped `db`: fresh copy per test, patched as the singleton.

@pytest.fixture(scope='session')
def _db_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a fully-provisioned SQLite database file (once per session).

    Runs the real production boot sequence — VersionedDatabaseService.provision()
    plus the policy seed (PolicyManager.apply_seed, as run.py does at boot) —
    against a temp file.  The result is a "golden" database that
    function-scoped fixtures copy cheaply.
    """
    import run as _run
    import services.database as _newdb
    from services.file_mapper_service import FileMapperService
    from services.policy_manager import PolicyManager
    from services.versioned_database_service import VersionedDatabaseService

    template_dir = tmp_path_factory.mktemp('db_template')
    template_path = str(template_dir / 'template.db')

    # The provisioner and the seed reach the DB through FileMapperService — the
    # provisioner resolves get_db_path() for its target and everything else from
    # that file's directory, the seed goes through the Database gateway. Point
    # get_db_path at the template file so the golden db lands there, in an empty
    # temp dir with no earlier database to copy forward (the fresh-install path).
    with patch.object(FileMapperService, 'get_db_path', return_value=Path(template_path)):
        _newdb.Database.close()  # drop any thread connection bound to another path
        VersionedDatabaseService().provision()
        # Mirror production boot (run.py): provisioning applies only static column
        # DEFAULTs, never value backfills, so the deterministic redesign-column
        # backfill runs as its own startup step. Without this the template
        # diverges from a real boot — valid_from / valid_to stay NULL where
        # production would have populated them.
        _run._backfill_redesign_columns()

        # Mirror boot: seed the flat policy table so gated tool calls on non-chat
        # channels (e.g. subconscious bash.* / timer) resolve to their real defaults
        # instead of an empty-table lazy 'ask'→deny. (PolicyManager.INTERNAL tools
        # bypass the gate entirely and carry no seed rows.)
        PolicyManager().apply_seed()

        # Flush WAL into main file so shutil.copy2 gets a self-contained copy
        _newdb.Database.conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _newdb.Database.close()
    return template_path


@pytest.fixture
def db(_db_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sqlite3.Connection]:
    """Fresh, fully-migrated SQLite database — one per test.

    Copies the session-scoped template, points the ``Database`` gateway
    (``FileMapperService.get_db_path``) at it, and drops any stale thread
    connection so every ``Database.conn()`` call — on-spine and off-spine alike
    — lands on this test's file. Yields the raw ``sqlite3.Connection`` for
    seeding data.

    Usage::

        def test_something(self, db):
            db.execute("INSERT INTO lists (id, name) VALUES ('l1', 'Groceries')")
            db.commit()
            result = my_service.get_list('Groceries')
            assert result['name'] == 'Groceries'
    """
    import services.database as _newdb
    from services.file_mapper_service import FileMapperService

    test_db_path = str(tmp_path / 'test.db')
    shutil.copy2(_db_template, test_db_path)

    # Point the Database gateway at this test's file (the default path resolves
    # through FileMapperService) so every Database.conn() call lands on it, then
    # drop any stale thread connection bound to another path.
    monkeypatch.setattr(FileMapperService, 'get_db_path', lambda *_: Path(test_db_path))
    # Redirect the telemetry snapshot to this test's tmp dir, next to the DB
    # patch above, so a heartbeat in a test never touches the real
    # data/telemetry.json.
    monkeypatch.setattr(FileMapperService, 'get_telemetry_json_path', lambda *_: tmp_path / 'telemetry.json')
    _newdb.Database.close()
    # Bind the connection getter onto Model — the boot step run.py runs once at
    # startup (``Database().bind()``). Repeated per test because each Database()
    # captures this test's patched path, so the getter must point at the current
    # test's file, not a prior test's or the real chalie.db.
    _newdb.Database().bind()

    # Invalidate the telemetry cache so the next read re-loads this test's
    # JSON file (the write path persists to the patched tmp location above).
    from services.telemetry_service import TelemetryService
    TelemetryService._cache = None
    # Clear the MCP connection map too — it is process-memory, not per-DB, so
    # one test's pings must never leak into the next.
    from services.mcp_client_service import McpClientService
    McpClientService._connected = {}

    conn = _newdb.Database.conn()
    try:
        yield conn
    finally:
        _newdb.Database.close()
        TelemetryService._cache = None
        McpClientService._connected = {}


#: Deliberately below MAX_CONTEXT_WINDOW (200_000) so a test asserting this
#: number proves it came from the provider row, not a silently-substituted cap.
SEEDED_CONTEXT_WINDOW = 120_000


@pytest.fixture
def chat_provider(db: sqlite3.Connection) -> sqlite3.Connection:
    """Seed a selected CHAT provider whose context window is already pinned.

    Every turn-driving test needs one. The context window is a column on
    ``providers``, so ``ProviderService`` reads it from the row rather than
    asking the transport client — which means a test that patches the client but
    seeds no provider has no window to be sized against, exactly like a real
    install with nothing configured. Patching the network boundary replaces what
    the provider *says*; it was never a substitute for the provider *existing*.

    The window is pre-pinned rather than probed so these tests stay hermetic:
    an unpinned row would send ``pin_context_window`` to the real host.
    """
    db.execute(
        "INSERT INTO providers (name, platform, model, host, context_window) "
        "VALUES ('test-chat-provider', 'ollama', 'test-model', "
        "'http://localhost:11434', ?)",
        (SEEDED_CONTEXT_WINDOW,),
    )
    row = db.execute(
        "SELECT id FROM providers WHERE name = 'test-chat-provider'"
    ).fetchone()
    db.execute(
        "INSERT INTO settings (key, value, value_type, description) "
        "VALUES ('selected_provider_id', ?, 'int', 'seeded by the chat_provider fixture') "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(row["id"]),),
    )
    db.commit()
    return db


# ── Vault backup isolation ────────────────────────────────────────
# Any test that initialises the real vault (directly or via /auth/register)
# writes a permanent vault_backup_*.json. Redirect the secure dir to a per-test
# temp path so backups never accumulate in the repo's data/secure/.

@pytest.fixture(autouse=True)
def _isolate_vault_backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from services.file_mapper_service import FileMapperService
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", tmp_path / "secure")


# Every settled turn on an in-scope channel offers itself to the memory-step
# service, which spawns a live background MessageProcessor. In the unit suite
# that thread outlives its test — consuming scripted provider responses,
# writing rows mid-teardown, and mutating per-channel gate state. The patch is
# session-scoped because fire-and-forget MP drive threads outlive their test:
# a per-test patch leaves teardown/setup gaps where a late settle would hit
# the real trigger against a torn-down DB. Feature tests re-arm the real
# trigger with the ``real_memory_step`` fixture.

_REAL_ON_SETTLE = None


@pytest.fixture(scope="session", autouse=True)
def _quiesce_memory_step() -> "Iterator[None]":
    global _REAL_ON_SETTLE
    from services.memory_step_service import MemoryStepService
    _REAL_ON_SETTLE = MemoryStepService.on_settle
    patch_ = pytest.MonkeyPatch()
    patch_.setattr(MemoryStepService, "on_settle", lambda self, mp: None)
    yield
    patch_.undo()


@pytest.fixture
def real_memory_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-arm the real settle trigger for memory-step feature tests."""
    from services.memory_step_service import MemoryStepService
    monkeypatch.setattr(MemoryStepService, "on_settle", _REAL_ON_SETTLE)


# The chat-history compactor hands every folded USER window to the
# user-synthesis generator, which spawns a live background MessageProcessor —
# the same leak shape as the memory step above, quiesced session-scoped for
# the same reason: no teardown/setup gap where a late real compaction could
# hit the real trigger. Feature tests re-arm with ``real_user_synthesis``.

_REAL_ON_COMPACTION = None


@pytest.fixture(scope="session", autouse=True)
def _quiesce_user_synthesis() -> "Iterator[None]":
    global _REAL_ON_COMPACTION
    from services.user_synthesis_generator import UserSynthesisGenerator
    _REAL_ON_COMPACTION = UserSynthesisGenerator.on_compaction
    patch_ = pytest.MonkeyPatch()
    patch_.setattr(
        UserSynthesisGenerator, "on_compaction", lambda self, channel, folded_block: None
    )
    yield
    patch_.undo()


@pytest.fixture
def real_user_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-arm the real compaction trigger for generator feature tests."""
    from services.user_synthesis_generator import UserSynthesisGenerator
    monkeypatch.setattr(UserSynthesisGenerator, "on_compaction", _REAL_ON_COMPACTION)


# Every settled turn kicks background speech pre-synthesis on a fire-and-forget
# daemon thread — the third instance of the leak shape above, and the one that
# corrupts rather than merely races: the thread records against
# ``voice_transcript.transcript_id``, a foreign key onto ``transcript(id)``, so
# a turn settled by one test writes after that test's rows are gone and SQLite
# raises IntegrityError inside a thread nobody joins. Session-scoped for the
# same reason as the two above. Pre-synthesis tests re-arm the real hook with
# ``real_voice_presynthesis``.

_REAL_VOICE_PRESYNTHESIS = None


@pytest.fixture(scope="session", autouse=True)
def _quiesce_voice_presynthesis() -> "Iterator[None]":
    global _REAL_VOICE_PRESYNTHESIS
    from controllers.message_processor import MessageProcessor
    _REAL_VOICE_PRESYNTHESIS = MessageProcessor._voice_presynthesis
    patch_ = pytest.MonkeyPatch()
    patch_.setattr(MessageProcessor, "_voice_presynthesis", lambda self: None)
    yield
    patch_.undo()


@pytest.fixture
def real_voice_presynthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-arm the real settle hook for pre-synthesis feature tests."""
    from controllers.message_processor import MessageProcessor
    monkeypatch.setattr(
        MessageProcessor, "_voice_presynthesis", _REAL_VOICE_PRESYNTHESIS
    )


# ── Non-DB mock fixtures ──────────────────────────────────────────

@pytest.fixture
def store() -> "Iterator[MemoryStore]":
    """Isolated MemoryStore — same implementation used in production.

    Patches both the canonical ``get_shared_store()`` in memory_store and the
    legacy ``MemoryClientService.create_connection()`` shim so every code path
    sees the same isolated instance.

    Yields:
        MemoryStore: A fresh, fully-functional in-process store instance.
    """
    from services.memory_store import MemoryStore
    _store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=_store):
        yield _store


@pytest.fixture
def authed_client(db: sqlite3.Connection) -> Iterator[tuple[object, sqlite3.Connection, object]]:
    """Flask test client with real blueprints registered, auth bypassed.

    Uses the real ``db`` fixture (which points the ``Database`` gateway at a
    per-test SQLite file), so Flask route handlers hit a real SQLite database.  The memory store is a
    real ``MemoryStore`` instance (not a ``MagicMock``), so route handlers that
    read or write store state work correctly in integration tests.

    Yields:
        tuple[FlaskClient, sqlite3.Connection, MemoryStore]: A 3-tuple of the
        Flask test client, the raw SQLite connection for seeding data, and the
        isolated in-process memory store.

    Usage::

        def test_endpoint(self, authed_client):
            client, db_conn, store = authed_client
            db_conn.execute("INSERT INTO ...")
            db_conn.commit()
            response = client.get('/health')
    """
    from api import create_app
    from services.memory_store import MemoryStore

    real_store = MemoryStore()

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.memory_store.get_shared_store', return_value=real_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=real_store):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield (client, db, real_store)


class _FixedEmbedder:
    """Deterministic stand-in for the embedding model: every text maps to the
    same 768-d unit vector, matching the vec-table dimension. Vector-lane hits
    all end up equidistant from the query, so ranking is decided by secondary
    keys (e.g. ``iteration``)."""

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        return [1.0] + [0.0] * 767


@pytest.fixture
def fixed_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin both embedding seams — indexing and recall — to one deterministic model."""
    emb = _FixedEmbedder()
    monkeypatch.setattr("services.embedding_service.get_embedding_service", lambda: emb)
    monkeypatch.setattr("services.memory_recall_service.get_embedding_service", lambda: emb)


class _LexicalEmbedder:
    """Deterministic stand-in whose vectors actually MOVE with the words in the
    text: each distinct token claims one of the 768 dimensions and the vector is
    L2-normalised, so vec0 distance falls as word overlap rises.

    Needed wherever a test must prove WHICH column the search key reads, or that
    ranking follows distance — neither is observable under
    :class:`_FixedEmbedder`, where every text is equidistant from every query.
    """

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        vec = [0.0] * 768
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(token.encode()) % 768] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


@pytest.fixture
def lexical_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin both embedding seams to a model whose distances discriminate by word
    overlap — the only way a recall test can tell one candidate from another."""
    emb = _LexicalEmbedder()
    monkeypatch.setattr("services.embedding_service.get_embedding_service", lambda: emb)
    monkeypatch.setattr("services.memory_recall_service.get_embedding_service", lambda: emb)
