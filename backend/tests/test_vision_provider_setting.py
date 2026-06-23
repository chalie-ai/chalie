import sqlite3
from typing import cast

import pytest

from services.database_service import DatabaseService
from services.provider_db_service import ProviderDbService

pytestmark = pytest.mark.unit


def _svc(db: sqlite3.Connection) -> ProviderDbService:
    import services.database_service as _db_mod
    return ProviderDbService(cast(DatabaseService, _db_mod._shared_db_service))


def _make_provider(db: sqlite3.Connection, name: str, vision: int) -> int:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO providers (name, platform, model, supports_vision) "
        "VALUES (?, 'ollama', 'llava', ?)",
        (name, vision),
    )
    db.commit()
    return cast(int, cur.lastrowid)


def test_explicit_vision_provider_resolves(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    pid = _make_provider(db, "vision-one", vision=1)
    svc.set_vision_provider(pid)
    got = svc.get_vision_provider()
    assert got is not None and got["id"] == pid
    status = svc.get_vision_provider_status()
    assert status["source"] == "explicit" and cast(dict[str, object], status["provider"])["id"] == pid


def test_explicit_nonvision_falls_through_to_none(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    pid = _make_provider(db, "no-vision", vision=0)
    svc.set_vision_provider(pid)
    # explicit id points at a non-vision provider → not usable
    assert svc.get_vision_provider() is None
    assert svc.get_vision_provider_status()["source"] == "none"


def test_auto_default_to_active_vision_provider(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    pid = _make_provider(db, "active-vision", vision=1)
    svc.set_selected_provider(pid)          # active provider supports vision
    # no explicit vision id set
    got = svc.get_vision_provider()
    assert got is not None and got["id"] == pid
    assert svc.get_vision_provider_status()["source"] == "auto"


def test_none_when_no_vision_anywhere(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    pid = _make_provider(db, "plain", vision=0)
    svc.set_selected_provider(pid)
    assert svc.get_vision_provider() is None
    assert svc.get_vision_provider_status()["source"] == "none"


def test_set_none_clears_setting(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    pid = _make_provider(db, "v", vision=1)
    svc.set_vision_provider(pid)
    svc.set_vision_provider(None)
    # explicit cleared; no active provider → none
    assert svc.get_vision_provider_status()["source"] in ("none", "auto")
