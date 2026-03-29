"""Unit tests for ImapCapability — credential, connect, and ingest."""

from __future__ import annotations

from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_CREDS = {
    "imap:host": "imap.gmail.com", "imap:port": "993",
    "imap:password": "pw", "imap:email": "u@gmail.com", "imap:tls": "1",
}


def _make():
    from capabilities.imap_capability.capability import ImapCapability
    return ImapCapability()


def _email_bytes(subject="Test", from_="John <john@example.com>",
                 to="jane@example.com", date="Mon, 28 Mar 2026 10:00:00 +0000",
                 msg_id="<123@example.com>"):
    msg = MIMEText("body", "plain")
    msg["From"] = from_
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = date
    msg["Message-ID"] = msg_id
    return msg.as_bytes()


# --- Identity & configure ---

@pytest.mark.unit
def test_identity():
    assert _make().get_id() == "imap"


@pytest.mark.unit
def test_configure_missing_fields():
    with pytest.raises(ValueError, match="email and password required"):
        _make().configure({"email": "u@gmail.com"})


@pytest.mark.unit
def test_configure_unsupported_domain():
    cap = _make()
    with patch.object(cap, "store_credential"):
        with pytest.raises(ValueError, match="Unsupported email provider"):
            cap.configure({"email": "u@unknown-xyz.com", "password": "pw"})


@pytest.mark.unit
def test_configure_known_provider():
    cap = _make()
    with patch.object(cap, "store_credential") as ms, \
         patch.object(cap, "connect", return_value=True):
        cap.configure({"email": "u@gmail.com", "password": "pw"})
    ms.assert_any_call("imap:host", "imap.gmail.com")


@pytest.mark.unit
def test_configure_connect_failure():
    cap = _make()
    with patch.object(cap, "store_credential"), \
         patch.object(cap, "connect", return_value=False), \
         patch.object(cap, "delete_credentials") as d:
        with pytest.raises(ValueError, match="Connection test failed"):
            cap.configure({"email": "u@gmail.com", "password": "bad"})
    d.assert_called_once()


@pytest.mark.unit
def test_connect_success():
    cap = _make()
    mc = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_CREDS.get), \
         patch("imapclient.IMAPClient", return_value=mc):
        assert cap.connect() is True


@pytest.mark.unit
def test_connect_failure():
    cap = _make()
    with patch.object(cap, "load_credential", side_effect=_CREDS.get), \
         patch("imapclient.IMAPClient", side_effect=Exception("refused")):
        assert cap.connect() is False


# --- parse_headers ---

@pytest.mark.unit
def test_parse_headers():
    from capabilities.imap_capability.capability import parse_headers
    r = parse_headers(42, _email_bytes())
    assert (r["uid"], r["from_addr"], r["subject"]) == (42, "john@example.com", "Test")
    msg = MIMEText("body", "plain")
    msg["From"] = "news@example.com"
    msg["List-Unsubscribe"] = "<mailto:unsub@example.com>"
    assert parse_headers(1, msg.as_bytes())["has_unsubscribe"] is True


# --- ingest ---

@pytest.mark.unit
def test_ingest_not_connected():
    cap = _make()
    assert cap.ingest() == []


@pytest.mark.unit
def test_ingest_fresh():
    cap = _make()
    cap._connected = True
    mc = MagicMock()
    mc.search.return_value = [100, 101]
    mc.fetch.return_value = {
        100: {b"RFC822.HEADER": _email_bytes(subject="A")},
        101: {b"RFC822.HEADER": _email_bytes(subject="B")},
    }
    creds = {**_CREDS, "imap:last_uid": None}
    with patch.object(cap, "load_credential", side_effect=creds.get), \
         patch("imapclient.IMAPClient", return_value=mc), \
         patch.object(cap, "store_credential") as sc:
        results = cap.ingest()
    assert [r["subject"] for r in results] == ["A", "B"]
    sc.assert_called_with("imap:last_uid", "101")


@pytest.mark.unit
def test_ingest_incremental():
    cap = _make()
    cap._connected = True
    mc = MagicMock()
    mc.search.return_value = [200, 201]
    mc.fetch.return_value = {
        200: {b"RFC822.HEADER": _email_bytes(subject="Old")},
        201: {b"RFC822.HEADER": _email_bytes(subject="New")},
    }
    creds = {**_CREDS, "imap:last_uid": "200"}
    with patch.object(cap, "load_credential", side_effect=creds.get), \
         patch("imapclient.IMAPClient", return_value=mc), \
         patch.object(cap, "store_credential"):
        results = cap.ingest()
    assert len(results) == 1


# --- classify_email ---

def _item(addr="a@b.com", subj="Hi", unsub=False, reply_to="", from_name=""):
    return {
        "has_unsubscribe": unsub, "from_addr": addr,
        "subject": subj, "in_reply_to": reply_to, "from_name": from_name,
    }


# Orthogonal unit vectors for deterministic mock anchors
_V_NOISE = np.array([1.0, 0.0, 0.0])
_V_ACTION = np.array([0.0, 1.0, 0.0])
_V_INFO = np.array([0.0, 0.0, 1.0])


def _route_embedding(text):
    """Map email text to a triage vector based on keywords."""
    t = text.lower()
    if any(w in t for w in ("invoice", "urgent", "deadline", "confirm", "approve")):
        return _V_ACTION
    if any(w in t for w in ("newsletter", "notification", "promo", "digest")):
        return _V_NOISE
    return _V_INFO


@pytest.fixture
def _mock_email_classify():
    """Full embedding mock: anchors + keyword-routed vectors."""
    import capabilities.imap_capability.capability as cap_mod
    old = cap_mod._anchor_cache
    cap_mod._anchor_cache = {
        "noise": _V_NOISE, "actionable": _V_ACTION, "informational": _V_INFO,
    }
    mock_svc = MagicMock()
    mock_svc.generate_embedding_np.side_effect = _route_embedding
    _p = "services.embedding_service.get_embedding_service"
    with patch(_p, return_value=mock_svc):
        yield mock_svc
    cap_mod._anchor_cache = old


@pytest.mark.unit
def test_classify_email_fast_path_unsubscribe():
    """Unsubscribe header is the only deterministic fast path."""
    from capabilities.imap_capability.capability import classify_email
    assert classify_email(_item(unsub=True)) == "noise"


@pytest.mark.unit
@pytest.mark.parametrize("item,expected", [
    (_item(addr="noreply@github.com", subj="New notification"), "noise"),
    (_item(subj="Weekly Newsletter Digest"), "noise"),
    (_item(subj="Invoice #123 Payment Due"), "actionable"),
    (_item(subj="Please confirm your attendance"), "actionable"),
    (_item(addr="colleague@work.com", subj="Meeting notes"), "informational"),
    (_item(subj="Project status update"), "informational"),
    # Thread replies go through embeddings — not the old in_reply_to fast path
    (_item(reply_to="<prev@ex>", subj="Re: Weekly Newsletter Digest"), "noise"),
    (_item(reply_to="<prev@ex>", addr="noreply@github.com",
           subj="Re: notification"), "noise"),
    (_item(reply_to="<prev@ex>",
           subj="Re: Please confirm your attendance"), "actionable"),
    (_item(reply_to="<prev@ex>", subj="Re: Project status update"), "informational"),
])
def test_classify_email_embedding(item, expected, _mock_email_classify):
    """Embedding path classifies by cosine similarity to anchors."""
    from capabilities.imap_capability.capability import classify_email
    assert classify_email(item) == expected


# --- understand ---

def _email_item(uid=1, msg_id="<m@ex>", subj="Hi", name="Bob", addr="bob@ex.com", unsub=False, reply=""):
    return {"uid": uid, "message_id": msg_id, "subject": subj, "from_name": name,
            "from_addr": addr, "has_unsubscribe": unsub, "in_reply_to": reply}


@pytest.mark.unit
def test_understand_empty_returns_empty():
    assert _make().understand([]) == []


@pytest.mark.unit
def test_understand_classifies_without_storing(_mock_email_classify):
    cap = _make()
    items = [
        _email_item(),
        _email_item(uid=2, msg_id="<m2@ex>", unsub=True),
        _email_item(uid=3, msg_id="<m3@ex>", reply="<prev@ex>"),
    ]
    result = cap.understand(items)
    assert result[0]["triage"] == "informational"
    assert result[1]["triage"] == "noise"
    assert result[2]["triage"] == "informational"  # thread replies use embeddings now
    assert result[2]["is_thread"] is True


# --- monitor ---

@pytest.mark.unit
def test_monitor_calls_ingest_understand():
    cap = _make()
    cap._connected = True
    with patch.object(cap, "ingest", return_value=[{"uid": 1}]) as ing, \
         patch.object(cap, "understand") as und:
        cap.monitor()
    ing.assert_called_once()
    und.assert_called_once()


@pytest.mark.unit
def test_monitor_skips_understand_when_empty():
    cap = _make()
    cap._connected = True
    with patch.object(cap, "ingest", return_value=[]), \
         patch.object(cap, "understand") as und:
        cap.monitor()
    und.assert_not_called()


# --- _imap_date ---

@pytest.mark.unit
def test_imap_date_conversion():
    from capabilities.imap_capability.capability import _imap_date
    assert _imap_date("2026-03-01") == "01-Mar-2026"
    assert _imap_date("2026-12-25") == "25-Dec-2026"
    assert _imap_date("bad-date") == "bad-date"  # passthrough on parse failure


# --- get_tools / imap_search_email ---

def _mock_imap_client(headers_by_uid):
    """Build a mock IMAPClient that returns the given headers."""
    mc = MagicMock()
    mc.search.return_value = list(headers_by_uid.keys())
    all_data = {uid: {b"RFC822.HEADER": hdr} for uid, hdr in headers_by_uid.items()}
    mc.fetch.side_effect = lambda uids, _: {u: all_data[u] for u in uids if u in all_data}
    return mc


_SEARCH_EMAILS = {
    1: _email_bytes(subject="Invoice #42", from_="Sarah <sarah@work.com>"),
    2: _email_bytes(subject="Weekly newsletter", from_="News Bot <news@letters.com>"),
    3: _email_bytes(subject="Re: Project plan", from_="Sarah <sarah@work.com>",
                    msg_id="<3@ex>"),
    4: _email_bytes(subject="Meeting notes", from_="Alex <alex@team.com>"),
}


@pytest.mark.unit
def test_get_tools_returns_all_tools():
    cap = _make()
    tools = cap.get_tools()
    names = [t["name"] for t in tools]
    assert "imap_send_email" in names
    assert "imap_search_email" in names
    assert "imap_read_email" in names
    for t in tools:
        assert "handler" in t
        assert "parameters" in t


@pytest.mark.unit
def test_search_by_sender(_mock_email_classify):
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"sender": "sarah"})
    mc.search.assert_called_once()
    criteria = mc.search.call_args[0][0]
    assert "FROM" in criteria and "sarah" in criteria


@pytest.mark.unit
def test_search_by_subject(_mock_email_classify):
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"subject": "invoice"})
    criteria = mc.search.call_args[0][0]
    assert "SUBJECT" in criteria and "invoice" in criteria


@pytest.mark.unit
def test_search_by_keyword(_mock_email_classify):
    """keyword param maps to IMAP TEXT criterion for full-text body search."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"keyword": "invoice"})
    criteria = mc.search.call_args[0][0]
    assert "TEXT" in criteria and "invoice" in criteria


@pytest.mark.unit
def test_search_by_date_from(_mock_email_classify):
    """date_from param maps to IMAP SINCE criterion with DD-Mon-YYYY format."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"date_from": "2026-03-01"})
    criteria = mc.search.call_args[0][0]
    assert "SINCE" in criteria and "01-Mar-2026" in criteria


@pytest.mark.unit
def test_search_by_date_to(_mock_email_classify):
    """date_to param maps to IMAP BEFORE criterion with DD-Mon-YYYY format."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"date_to": "2026-03-28"})
    criteria = mc.search.call_args[0][0]
    assert "BEFORE" in criteria and "28-Mar-2026" in criteria


@pytest.mark.unit
def test_search_unanswered(_mock_email_classify):
    """unanswered=True adds UNANSWERED to IMAP criteria."""
    cap = _make()
    tools = cap.get_tools()
    handler = next(
        t for t in tools if t["name"] == "imap_search_email"
    )["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"sender": "sarah", "unanswered": True})
    criteria = mc.search.call_args[0][0]
    assert "UNANSWERED" in criteria
    assert "FROM" in criteria


@pytest.mark.unit
def test_search_combined_criteria(_mock_email_classify):
    """Multiple params combine into a single IMAP criteria list."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        handler("t1", {"sender": "sarah", "keyword": "budget",
                        "date_from": "2026-03-01"})
    criteria = mc.search.call_args[0][0]
    assert "FROM" in criteria
    assert "TEXT" in criteria
    assert "SINCE" in criteria


@pytest.mark.unit
def test_search_by_triage(_mock_email_classify):
    """Triage filtering happens client-side after IMAP fetch."""
    cap = _make()
    # Only return the actionable email (Invoice has "invoice" in subject)
    mc = _mock_imap_client({1: _SEARCH_EMAILS[1]})
    with patch.object(cap, "_open_client", return_value=mc):
        result = handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
        result = handler("t1", {"triage": "actionable"})
    assert all(e["triage"] == "actionable" for e in result["emails"])


@pytest.mark.unit
def test_search_with_limit(_mock_email_classify):
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client(_SEARCH_EMAILS)
    with patch.object(cap, "_open_client", return_value=mc):
        result = handler("t1", {"limit": 1})
    # Server gets the limit applied to UIDs
    assert len(result["emails"]) <= 1


@pytest.mark.unit
def test_search_no_results():
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    mc = _mock_imap_client({})
    mc.search.return_value = []
    with patch.object(cap, "_open_client", return_value=mc):
        result = handler("t1", {"sender": "nobody"})
    assert result["count"] == 0
    assert result["emails"] == []


@pytest.mark.unit
def test_search_error_resilience():
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_search_email")["handler"]
    with patch.object(cap, "_open_client", return_value=None):
        result = handler("t1", {})
    assert "error" in result


# --- _inject_inbox_hint ---

# Unseen emails for hint tests: 1 actionable (Invoice from Sarah), 1 informational (Meeting from Alex)
_HINT_EMAILS = {
    1: _email_bytes(subject="Invoice #42", from_="Sarah <sarah@work.com>"),
    4: _email_bytes(subject="Meeting notes", from_="Alex <alex@team.com>"),
}


@pytest.mark.unit
def test_inject_inbox_hint_calls_world_state(_mock_email_classify):
    cap = _make()
    mc = _mock_imap_client(_HINT_EMAILS)
    ws_mock = MagicMock()
    with patch.object(cap, "_open_client", return_value=mc), \
         patch("services.world_state_service.WorldStateService", return_value=ws_mock):
        cap._inject_inbox_hint()
    ws_mock.notify_external_signal.assert_called_once()
    call_kw = ws_mock.notify_external_signal.call_args
    assert call_kw.kwargs["signal_type"] == "capability:imap"
    assert call_kw.kwargs["source"] == "imap"


@pytest.mark.unit
def test_inject_inbox_hint_content_has_counts(_mock_email_classify):
    cap = _make()
    mc = _mock_imap_client(_HINT_EMAILS)
    ws_mock = MagicMock()
    with patch.object(cap, "_open_client", return_value=mc), \
         patch("services.world_state_service.WorldStateService", return_value=ws_mock):
        cap._inject_inbox_hint()
    content = ws_mock.notify_external_signal.call_args.kwargs["content"]
    assert "1 actionable" in content
    assert "1 informational" in content
    assert "Sarah" in content  # top actionable sender


@pytest.mark.unit
def test_inject_inbox_hint_high_activation_when_actionable(_mock_email_classify):
    cap = _make()
    mc = _mock_imap_client(_HINT_EMAILS)
    ws_mock = MagicMock()
    with patch.object(cap, "_open_client", return_value=mc), \
         patch("services.world_state_service.WorldStateService", return_value=ws_mock):
        cap._inject_inbox_hint()
    assert ws_mock.notify_external_signal.call_args.kwargs["activation_energy"] == 0.8


@pytest.mark.unit
def test_inject_inbox_hint_skips_when_no_unseen():
    cap = _make()
    mc = _mock_imap_client({})
    mc.search.return_value = []
    with patch.object(cap, "_open_client", return_value=mc):
        cap._inject_inbox_hint()  # no WorldState call, just returns


@pytest.mark.unit
def test_inject_inbox_hint_skips_when_no_connection():
    cap = _make()
    with patch.object(cap, "_open_client", return_value=None):
        cap._inject_inbox_hint()  # should not raise


@pytest.mark.unit
def test_monitor_calls_inject_inbox_hint():
    cap = _make()
    cap._connected = True
    with patch.object(cap, "ingest", return_value=[]), \
         patch.object(cap, "_inject_inbox_hint") as hint_mock:
        cap.monitor()
    hint_mock.assert_called_once()


# --- _first_connect_greeting ---

def _setup_greeting_db():
    """Create an in-memory SQLite with the scheduled_items table."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE scheduled_items (
        id TEXT PRIMARY KEY, item_type TEXT, message TEXT, due_at TEXT,
        status TEXT, topic TEXT, source TEXT, external_uid TEXT,
        hidden INTEGER, created_at TEXT, recurrence TEXT, is_prompt INTEGER
    )""")
    conn.commit()
    return conn


@pytest.mark.unit
def test_first_connect_greeting_fires():
    """First call creates a greeting notification."""
    cap = _make()
    conn = _setup_greeting_db()
    db_mock = MagicMock()
    db_mock.connection.return_value.__enter__ = lambda s: conn
    db_mock.connection.return_value.__exit__ = MagicMock(return_value=False)
    items = [
        {"triage": "actionable", "uid": 1},
        {"triage": "informational", "uid": 2},
        {"triage": "noise", "uid": 3},
    ]
    with patch("services.database_service.get_shared_db_service", return_value=db_mock):
        cap._first_connect_greeting(items)
    row = conn.execute("SELECT message FROM scheduled_items WHERE external_uid = 'imap:greeting'").fetchone()
    assert row is not None
    assert "Email connected!" in row[0]
    assert "1 actionable" in row[0]
    assert "1 informational" in row[0]
    conn.close()


@pytest.mark.unit
def test_first_connect_greeting_skips_duplicate():
    """Second call does not insert a duplicate greeting."""
    cap = _make()
    conn = _setup_greeting_db()
    conn.execute(
        "INSERT INTO scheduled_items (id, item_type, message, due_at, status, "
        "topic, source, external_uid, hidden, created_at) "
        "VALUES ('x', 'notification', 'old', '2026-01-01', 'pending', "
        "'email', 'imap', 'imap:greeting', 0, '2026-01-01')"
    )
    conn.commit()
    db_mock = MagicMock()
    db_mock.connection.return_value.__enter__ = lambda s: conn
    db_mock.connection.return_value.__exit__ = MagicMock(return_value=False)
    with patch("services.database_service.get_shared_db_service", return_value=db_mock):
        cap._first_connect_greeting([{"triage": "actionable", "uid": 1}])
    rows = conn.execute("SELECT COUNT(*) FROM scheduled_items WHERE external_uid = 'imap:greeting'").fetchone()
    assert rows[0] == 1  # still just the original
    conn.close()


@pytest.mark.unit
def test_first_connect_greeting_survives_exception():
    """DB failure doesn't crash monitor()."""
    cap = _make()
    with patch("services.database_service.get_shared_db_service", side_effect=Exception("db down")):
        cap._first_connect_greeting([{"triage": "actionable", "uid": 1}])  # should not raise


@pytest.mark.unit
def test_monitor_calls_greeting_on_items():
    """monitor() calls _first_connect_greeting when items are returned."""
    cap = _make()
    cap._connected = True
    items = [{"uid": 1}]
    with patch.object(cap, "ingest", return_value=items), \
         patch.object(cap, "understand"), \
         patch.object(cap, "_first_connect_greeting") as greet, \
         patch.object(cap, "_inject_inbox_hint"):
        cap.monitor()
    greet.assert_called_once_with(items)


@pytest.mark.unit
def test_monitor_skips_greeting_when_empty():
    """monitor() does NOT call _first_connect_greeting when ingest returns empty."""
    cap = _make()
    cap._connected = True
    with patch.object(cap, "ingest", return_value=[]), \
         patch.object(cap, "_first_connect_greeting") as greet, \
         patch.object(cap, "_inject_inbox_hint"):
        cap.monitor()
    greet.assert_not_called()


# --- SMTP send ---

_SMTP_CREDS = {
    **_CREDS,
    "smtp:host": "smtp.gmail.com", "smtp:port": "465", "smtp:tls": "1",
}


@pytest.mark.unit
def test_configure_stores_smtp_settings():
    """configure() persists SMTP host/port/tls from autodiscovery."""
    cap = _make()
    stored = {}
    with patch.object(cap, "store_credential", side_effect=lambda k, v: stored.__setitem__(k, v)), \
         patch.object(cap, "connect", return_value=True):
        cap.configure({"email": "u@gmail.com", "password": "pw"})
    assert stored["smtp:host"] == "smtp.gmail.com"
    assert stored["smtp:port"] == "465"
    assert stored["smtp:tls"] == "1"


@pytest.mark.unit
def test_open_smtp_ssl():
    """_open_smtp() uses SMTP_SSL when tls=1."""
    cap = _make()
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch("smtplib.SMTP_SSL", return_value=smtp_mock) as ssl_cls:
        conn = cap._open_smtp()
    ssl_cls.assert_called_once_with("smtp.gmail.com", 465, timeout=30)
    smtp_mock.login.assert_called_once_with("u@gmail.com", "pw")
    assert conn is smtp_mock


@pytest.mark.unit
def test_open_smtp_starttls():
    """_open_smtp() uses SMTP + starttls when tls=0."""
    cap = _make()
    creds = {**_SMTP_CREDS, "smtp:port": "587", "smtp:tls": "0"}
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=creds.get), \
         patch("smtplib.SMTP", return_value=smtp_mock) as plain_cls:
        conn = cap._open_smtp()
    plain_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
    smtp_mock.starttls.assert_called_once()
    smtp_mock.login.assert_called_once()
    assert conn is smtp_mock


@pytest.mark.unit
def test_open_smtp_returns_none_on_failure():
    """_open_smtp() returns None on connection failure."""
    cap = _make()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch("smtplib.SMTP_SSL", side_effect=Exception("refused")):
        assert cap._open_smtp() is None


@pytest.mark.unit
def test_send_email_success():
    """_send_email() sends and returns success dict."""
    cap = _make()
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch.object(cap, "_open_smtp", return_value=smtp_mock):
        result = cap._send_email("bob@test.com", "Hi Bob", "Hello there")
    assert result["success"] is True
    assert result["to"] == "bob@test.com"
    smtp_mock.send_message.assert_called_once()
    smtp_mock.quit.assert_called_once()


@pytest.mark.unit
def test_send_email_reply_headers():
    """_send_email() sets In-Reply-To and References for replies."""
    cap = _make()
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch.object(cap, "_open_smtp", return_value=smtp_mock):
        cap._send_email("bob@test.com", "Re: Hi", "Reply body",
                        in_reply_to="<orig@example.com>")
    sent_msg = smtp_mock.send_message.call_args[0][0]
    assert sent_msg["In-Reply-To"] == "<orig@example.com>"
    assert sent_msg["References"] == "<orig@example.com>"


@pytest.mark.unit
def test_send_email_smtp_failure():
    """_send_email() returns error when SMTP connection fails."""
    cap = _make()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch.object(cap, "_open_smtp", return_value=None):
        result = cap._send_email("bob@test.com", "Hi", "Body")
    assert "error" in result


@pytest.mark.unit
def test_send_tool_validates_required_fields():
    """imap_send_email tool rejects missing to/subject/body."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_send_email")["handler"]
    assert "error" in handler("t1", {"to": "bob@test.com", "subject": "Hi"})
    assert "error" in handler("t1", {"to": "bob@test.com", "body": "text"})
    assert "error" in handler("t1", {"subject": "Hi", "body": "text"})


@pytest.mark.unit
def test_send_tool_calls_send_email():
    """imap_send_email tool delegates to _send_email."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_send_email")["handler"]
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch.object(cap, "_open_smtp", return_value=smtp_mock):
        result = handler("t1", {"to": "bob@test.com", "subject": "Hi", "body": "Hello"})
    assert result["success"] is True


# --- send_email name auto-resolution ---

@pytest.mark.unit
def test_send_tool_resolves_name_single_match():
    """When 'to' has no @, resolve via ContactResolver — single match sends directly."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_send_email")["handler"]
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch.object(cap, "_open_smtp", return_value=smtp_mock), \
         patch("capabilities.contact_resolver.resolve",
               return_value=[{"email": "sarah@work.com", "name": "Sarah Chen"}]):
        result = handler("t1", {"to": "Sarah", "subject": "Hi", "body": "Hello"})
    assert result["success"] is True
    assert result["resolved_from"] == "Sarah"
    sent_msg = smtp_mock.send_message.call_args[0][0]
    assert sent_msg["To"] == "sarah@work.com"


@pytest.mark.unit
def test_send_tool_resolves_name_multiple_matches():
    """When resolve returns multiple contacts, return disambiguation."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_send_email")["handler"]
    with patch("capabilities.contact_resolver.resolve",
               return_value=[
                   {"email": "sarah@work.com", "name": "Sarah Chen"},
                   {"email": "sarah@personal.com", "name": "Sarah Jones"},
               ]):
        result = handler("t1", {"to": "Sarah", "subject": "Hi", "body": "Hello"})
    assert result["error"] == "ambiguous_recipient"
    assert len(result["candidates"]) == 2


@pytest.mark.unit
def test_send_tool_resolves_name_no_matches():
    """When resolve returns nothing, return unknown_recipient error."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_send_email")["handler"]
    with patch("capabilities.contact_resolver.resolve", return_value=[]):
        result = handler("t1", {"to": "Nobody", "subject": "Hi", "body": "Hello"})
    assert result["error"] == "unknown_recipient"


@pytest.mark.unit
def test_send_tool_email_address_bypasses_resolve():
    """When 'to' contains @, skip resolution entirely."""
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_send_email")["handler"]
    smtp_mock = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_SMTP_CREDS.get), \
         patch.object(cap, "_open_smtp", return_value=smtp_mock), \
         patch("capabilities.contact_resolver.resolve") as resolve_mock:
        result = handler("t1", {"to": "bob@test.com", "subject": "Hi", "body": "Hello"})
    resolve_mock.assert_not_called()
    assert result["success"] is True
    assert "resolved_from" not in result


@pytest.mark.unit
def test_act_routes_send_email():
    """act('send_email', ...) delegates to _send_email."""
    cap = _make()
    with patch.object(cap, "_send_email", return_value={"success": True}) as m:
        result = cap.act("send_email", {"to": "a@b.com", "subject": "s", "body": "b"})
    m.assert_called_once()
    assert result["success"] is True


@pytest.mark.unit
def test_act_unknown_action():
    """act() returns error for unknown actions."""
    cap = _make()
    result = cap.act("delete_everything", {})
    assert "error" in result


# --- extract_body + imap_read_email ---

def _build_email(body="Hello", ct="plain", headers=True):
    """Build RFC822 email bytes with a body."""
    msg = MIMEText(body, ct, "utf-8")
    if headers:
        msg["From"] = "Sarah <sarah@work.com>"
        msg["To"] = "me@example.com"
        msg["Subject"] = "Test email"
        msg["Date"] = "Mon, 28 Mar 2026 10:00:00 +0000"
        msg["Message-ID"] = "<test@example.com>"
    return msg.as_bytes()


@pytest.mark.unit
def test_extract_body_plain_and_truncation():
    from capabilities.imap_capability.capability import extract_body
    assert "Hello world" in extract_body(_build_email("Hello world"))
    result = extract_body(_build_email("x" * 5000), max_chars=100)
    assert "[truncated]" in result and len(result) < 200


@pytest.mark.unit
def test_extract_body_multipart_and_html_fallback():
    from capabilities.imap_capability.capability import extract_body
    from email.mime.multipart import MIMEMultipart
    # Multipart: prefers plain
    msg = MIMEMultipart("alternative")
    msg.attach(MIMEText("Plain ver", "plain", "utf-8"))
    msg.attach(MIMEText("<p>HTML</p>", "html", "utf-8"))
    result = extract_body(msg.as_bytes())
    assert "Plain ver" in result and "<p>" not in result
    # HTML-only fallback
    msg2 = MIMEMultipart("alternative")
    msg2.attach(MIMEText("<p>HTML <b>bold</b></p>", "html", "utf-8"))
    result2 = extract_body(msg2.as_bytes())
    assert "HTML bold" in result2 and "<p>" not in result2


@pytest.mark.unit
def test_read_tool_success():
    cap = _make()
    raw = _build_email("Important content here")
    mc = MagicMock()
    mc.fetch.return_value = {42: {b"RFC822": raw}}
    with patch.object(cap, "_open_client", return_value=mc):
        handler = next(t for t in cap.get_tools() if t["name"] == "imap_read_email")["handler"]
        result = handler("t1", {"uid": 42})
    assert result["uid"] == 42 and "Important content here" in result["body"]
    assert result["from_addr"] == "sarah@work.com"


@pytest.mark.unit
def test_read_tool_error_cases():
    cap = _make()
    handler = next(t for t in cap.get_tools() if t["name"] == "imap_read_email")["handler"]
    assert "error" in handler("t1", {})  # missing uid
    with patch.object(cap, "_open_client", return_value=None):
        assert "error" in handler("t1", {"uid": 42})  # no connection


# --- imap_mark_read tool ---


@pytest.mark.unit
def test_mark_read_single_uid():
    """mark_read sets \\Seen flag via add_flags for a single UID."""
    cap = _make()
    mc = MagicMock()
    with patch.object(cap, "_open_client", return_value=mc):
        handler = next(
            t for t in cap.get_tools() if t["name"] == "imap_mark_read"
        )["handler"]
        result = handler("t1", {"uid": 42})
    assert result["success"] is True
    assert result["uids"] == [42]
    mc.select_folder.assert_called_once_with("INBOX", readonly=False)
    mc.add_flags.assert_called_once_with([42], [b"\\Seen"])
    mc.logout.assert_called_once()


@pytest.mark.unit
def test_mark_read_multiple_uids():
    """mark_read handles a list of UIDs."""
    cap = _make()
    mc = MagicMock()
    with patch.object(cap, "_open_client", return_value=mc):
        handler = next(
            t for t in cap.get_tools() if t["name"] == "imap_mark_read"
        )["handler"]
        result = handler("t1", {"uids": [10, 20, 30]})
    assert result["success"] is True
    assert result["uids"] == [10, 20, 30]
    mc.add_flags.assert_called_once_with([10, 20, 30], [b"\\Seen"])


@pytest.mark.unit
def test_mark_read_error_no_uid():
    """mark_read returns error when no uid or uids provided."""
    cap = _make()
    handler = next(
        t for t in cap.get_tools() if t["name"] == "imap_mark_read"
    )["handler"]
    assert "error" in handler("t1", {})


@pytest.mark.unit
def test_mark_read_error_no_connection():
    """mark_read returns error when IMAP connection fails."""
    cap = _make()
    with patch.object(cap, "_open_client", return_value=None):
        handler = next(
            t for t in cap.get_tools() if t["name"] == "imap_mark_read"
        )["handler"]
        result = handler("t1", {"uid": 42})
    assert "error" in result


@pytest.mark.unit
def test_mark_read_server_error():
    """mark_read returns error when add_flags raises."""
    cap = _make()
    mc = MagicMock()
    mc.add_flags.side_effect = Exception("Server refused")
    with patch.object(cap, "_open_client", return_value=mc):
        handler = next(
            t for t in cap.get_tools() if t["name"] == "imap_mark_read"
        )["handler"]
        result = handler("t1", {"uid": 42})
    assert "error" in result and "Server refused" in result["error"]


# --- imap_snooze ---

def _get_snooze_handler(cap):
    """Return the imap_snooze tool handler from a capability instance."""
    return next(
        t for t in cap.get_tools() if t["name"] == "imap_snooze"
    )["handler"]


@pytest.mark.unit
def test_snooze_missing_params():
    """snooze returns error when uid or snooze_until missing."""
    cap = _make()
    handler = _get_snooze_handler(cap)
    assert "error" in handler("t1", {"uid": 42})
    assert "error" in handler("t1", {"snooze_until": "2026-04-01T09:00:00+00:00"})
    assert "error" in handler("t1", {})


@pytest.mark.unit
def test_snooze_invalid_uid():
    """snooze returns error for non-integer uid."""
    cap = _make()
    handler = _get_snooze_handler(cap)
    result = handler("t1", {
        "uid": "abc", "snooze_until": "2026-04-01T09:00:00+00:00",
    })
    assert "error" in result


@pytest.mark.unit
def test_snooze_invalid_datetime():
    """snooze returns error for unparseable snooze_until."""
    cap = _make()
    handler = _get_snooze_handler(cap)
    result = handler("t1", {"uid": 42, "snooze_until": "not-a-date"})
    assert "error" in result and "ISO 8601" in result["error"]


@pytest.mark.unit
def test_snooze_no_connection():
    """snooze returns error when IMAP connection fails."""
    cap = _make()
    with patch.object(cap, "_open_client", return_value=None):
        handler = _get_snooze_handler(cap)
        result = handler("t1", {
            "uid": 42,
            "snooze_until": "2026-04-01T09:00:00+00:00",
        })
    assert "error" in result and "Cannot connect" in result["error"]


@pytest.mark.unit
def test_snooze_email_not_found():
    """snooze returns error when UID doesn't exist on server."""
    cap = _make()
    mc = MagicMock()
    mc.fetch.return_value = {}
    with patch.object(cap, "_open_client", return_value=mc):
        handler = _get_snooze_handler(cap)
        result = handler("t1", {
            "uid": 999,
            "snooze_until": "2026-04-01T09:00:00+00:00",
        })
    assert "error" in result and "not found" in result["error"]
    mc.logout.assert_called_once()


@pytest.mark.unit
def test_snooze_creates_scheduled_item():
    """snooze creates a prompt scheduled_item with correct fields."""
    import sqlite3
    cap = _make()
    mc = MagicMock()
    mc.fetch.return_value = {
        42: {b"RFC822.HEADER": _email_bytes(subject="Invoice Q1")}
    }

    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE scheduled_items (
        id TEXT PRIMARY KEY, item_type TEXT, message TEXT, due_at TEXT,
        status TEXT, topic TEXT, source TEXT, external_uid TEXT,
        hidden INTEGER, created_at TEXT, is_prompt INTEGER,
        recurrence TEXT, window_start TEXT, window_end TEXT,
        created_by_session TEXT, group_id TEXT, last_fired_at TEXT,
        metadata TEXT)""")
    conn.commit()
    mock_db = MagicMock()
    mock_db.connection.return_value.__enter__ = lambda s: conn
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(cap, "_open_client", return_value=mc), \
         patch("services.database_service.get_shared_db_service",
               return_value=mock_db):
        handler = _get_snooze_handler(cap)
        result = handler("t1", {
            "uid": 42,
            "snooze_until": "2026-04-01T09:00:00+00:00",
            "reason": "reply with Q1 numbers",
        })

    assert result["success"] is True
    assert result["email_uid"] == 42
    assert result["subject"] == "Invoice Q1"
    assert "snooze_id" in result

    row = conn.execute("SELECT * FROM scheduled_items").fetchone()
    assert row is not None
    # item_type is 'prompt'
    assert row[1] == "prompt"
    # source is 'imap:snooze'
    assert row[6] == "imap:snooze"
    # external_uid is 'snooze:42'
    assert row[7] == "snooze:42"
    # is_prompt is 1
    assert row[10] == 1
    # message contains the reason
    assert "Q1 numbers" in row[2]
    conn.close()


@pytest.mark.unit
def test_snooze_prevents_duplicate():
    """snooze returns error if the same UID already has a pending snooze."""
    import sqlite3
    cap = _make()
    mc = MagicMock()
    mc.fetch.return_value = {
        42: {b"RFC822.HEADER": _email_bytes(subject="Invoice")}
    }

    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE scheduled_items (
        id TEXT PRIMARY KEY, item_type TEXT, message TEXT, due_at TEXT,
        status TEXT, topic TEXT, source TEXT, external_uid TEXT,
        hidden INTEGER, created_at TEXT, is_prompt INTEGER,
        recurrence TEXT, window_start TEXT, window_end TEXT,
        created_by_session TEXT, group_id TEXT, last_fired_at TEXT,
        metadata TEXT)""")
    conn.execute(
        "INSERT INTO scheduled_items (id, item_type, message, due_at, "
        "status, source, external_uid, is_prompt) "
        "VALUES ('existing', 'prompt', 'msg', '2026-04-01', 'pending', "
        "'imap:snooze', 'snooze:42', 1)"
    )
    conn.commit()
    mock_db = MagicMock()
    mock_db.connection.return_value.__enter__ = lambda s: conn
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(cap, "_open_client", return_value=mc), \
         patch("services.database_service.get_shared_db_service",
               return_value=mock_db):
        handler = _get_snooze_handler(cap)
        result = handler("t1", {
            "uid": 42,
            "snooze_until": "2026-04-02T09:00:00+00:00",
        })

    assert result.get("error") == "duplicate"
    conn.close()
