"""Feature test: ``/api/system/observability/token-usage`` — the READ path.

Where ``test_llm_call_log_usage_type.py`` exercises *billing* — which bucket
``llm_call_log.type`` assigns per row — this test exercises the inverse
direction: given a handful of seeded rows, what does ``LlmLogService.token_usage``
— exposed verbatim over the real Flask ``ObservabilityTokenUsageResource`` —
actually return? Six claims, one per thing the Brain UI blindly trusts, plus
one on the route's own input contract:

1. entries are grouped by ``(bucket, type, model, provider)`` — never collapsed;
2. ``type=foreground`` narrows the response to foreground rows only;
3. ``type=background`` narrows to background rows only;
4. unfiltered brings both types into the response;
5. summary totals exactly reproduce the seeded numbers;
6. a deliberately old row is genuinely excluded by the window bound;
7. an unrecognised ``window`` value is rejected with 422, not a 500.

Everything runs against a real migrated SQLite database (the session-scoped
template + function-scoped copy behind ``conftest.authed_client``), with
``validate_session`` patched off — auth is plumbing, not the claim under test.
Nothing about the service, its SQL, or the model is mocked inside the test.

Seed timestamps are computed by SQLite's own ``datetime('now', ?)`` — the
identical function ``LlmLogService._usage_buckets`` uses for its window bound
(``llm_call_log.created_at`` even *defaults* to ``datetime('now')`` in
``schema.sql``) — so the seeds ride the same clock the query does. There is no
absolute date literal anywhere in this file: a fixed calendar date would drift
out of the ``day``/``week``/``month`` windows as real time passes and start
failing on a schedule unrelated to any defect in the code under test.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import cast

import pytest
from flask.testing import FlaskClient

pytestmark = pytest.mark.unit

# -----------------------------------------------------------------------
# Seed data. Every row's created_at is computed by SQLite itself via
# datetime('now', <offset>) at INSERT time -- the same primitive
# `_usage_buckets` uses for its window bound -- so seeding and querying share
# one clock and the test never depends on the wall-clock date it happens to
# run on.
# -----------------------------------------------------------------------

_SQL_INSERT = (
    "INSERT INTO llm_call_log "
    "(type, provider, model, tokens_input, tokens_output, "
    "tokens_cache_read, tokens_cache_create, tokens_thinking, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', ?))"
)

# "-5 minutes" sits comfortably inside the tightest window exercised (`day`,
# a -1 day bound) with room to spare even across an hour-bucket boundary.
# "-365 days" sits comfortably outside the widest finite window exercised
# (`month`, a -30 day bound).
_RECENT_OFFSET = "-5 minutes"
_OUTLIER_OFFSET = "-365 days"

_SEED_ROWS = (
    # foreground, provider-x, alpha -- recent.
    ("foreground", "provider-x", "alpha", 100, 20, 10, 2, 5, _RECENT_OFFSET),
    # background, provider-x, alpha -- SAME bucket/model/provider, DIFFERENT type.
    # This is the GROUP-BY collision test: without `type` in the GROUP BY,
    # this row and the one above would fold into ONE row with summed totals
    # -- the exact wrong response.
    ("background", "provider-x", "alpha", 40, 8, 4, 1, 2, _RECENT_OFFSET),
    # foreground, provider-y, beta -- third producer, third model, still recent.
    ("foreground", "provider-y", "beta", 200, 40, 20, 4, 10, _RECENT_OFFSET),
    # YEAR-old outlier with astronomical counts -- excluded from day/week/
    # month. If the window filter is broken, this single row poisons the sum
    # by +1_675_000, impossible to hide behind rounding.
    (
        "foreground", "provider-x", "alpha", 1_000_000, 500_000,
        100_000, 50_000, 25_000, _OUTLIER_OFFSET,
    ),
)

# Hand-computed expectations, derived solely from the seed rows above.
_FOREGROUND_RECENT = 100 + 20 + 10 + 2 + 5 + 200 + 40 + 20 + 4 + 10  # 411
_BACKGROUND_RECENT = 40 + 8 + 4 + 1 + 2  # 55
_RECENT_TOTAL = _FOREGROUND_RECENT + _BACKGROUND_RECENT  # 466
_OUTLIER_TOTAL = 1_000_000 + 500_000 + 100_000 + 50_000 + 25_000  # 1_675_000
_LIFETIME_TOTAL = _RECENT_TOTAL + _OUTLIER_TOTAL


@pytest.fixture
def seeded_client(authed_client: tuple[object, sqlite3.Connection, object]) -> FlaskClient:
    """``authed_client``'s Flask test client, with the ledger seeded.

    Reuses the shared ``authed_client`` fixture from ``conftest.py`` rather
    than a bespoke duplicate -- it already patches auth, wires a real
    ``MemoryStore``, and hands back the same per-test SQLite connection the
    ``db`` fixture backs. Clears any pre-existing ``llm_call_log`` rows (the
    boot template seeds none, but this keeps the fixture correct even if that
    changes) then inserts the deterministic spread above.
    """
    client, db_conn, *_ = authed_client
    db_conn.execute("DELETE FROM llm_call_log")
    for values in _SEED_ROWS:
        db_conn.execute(_SQL_INSERT, values)
    db_conn.commit()
    return cast(FlaskClient, client)


class TestTokenUsageReadPath:
    """Token-usage read path: bucketing, filtering and window bounds."""

    # ------------------------------------------------------------------
    # Claim (a): grouping.
    # ------------------------------------------------------------------

    def test_entries_grouped_by_bucket_type_model_provider(self, seeded_client: FlaskClient) -> None:
        """Two rows sharing (bucket=recent, model=alpha, provider=x) yet
        differing in type MUST appear as TWO separate entries. Dropping
        ``type`` from the GROUP BY would merge them silently and half the
        totals in the UI become wrong.

        Counts asserted: tokens_input=100 for foreground, 40 for background.
        Any merge would give one big row with summed counts — this assertion
        fires on exactly that.
        """
        resp = seeded_client.get("/api/system/observability/token-usage?window=day")
        assert resp.status_code == 200, f"unexpected HTTP {resp.status_code}: {resp.data[:200]}"

        body = json.loads(resp.data)
        assert body["window"] == "day"

        same_combo = [
            e for e in body["entries"]
            if e["provider"] == "provider-x" and e["model"] == "alpha"
        ]
        assert len(same_combo) == 2, (
            f"expected two distinct entries for (bucket,provider-x,alpha) "
            f"across foreground+background, got {[e['type'] for e in same_combo]}"
        )
        types_seen = sorted({e["type"] for e in same_combo})
        assert types_seen == ["background", "foreground"], types_seen

        foreground_row = next(e for e in same_combo if e["type"] == "foreground")
        background_row = next(e for e in same_combo if e["type"] == "background")
        assert foreground_row["tokens_input"] == 100
        assert foreground_row["tokens_output"] == 20
        assert background_row["tokens_input"] == 40
        assert background_row["tokens_output"] == 8

    # ------------------------------------------------------------------
    # Claims (b), (c): type filters isolate rows cleanly.
    # ------------------------------------------------------------------

    def test_foreground_filter_excludes_background_rows(self, seeded_client: FlaskClient) -> None:
        """``type=foreground`` must not surface any background entry."""
        resp = seeded_client.get("/api/system/observability/token-usage?window=day&type=foreground")
        assert resp.status_code == 200
        body = json.loads(resp.data)

        types_in_response = {e["type"] for e in body["entries"]}
        assert types_in_response <= {"foreground"}, (
            f"type=foreground still surfaced non-foreground entries: {types_in_response}"
        )

    def test_background_filter_excludes_foreground_rows(self, seeded_client: FlaskClient) -> None:
        """``type=background`` must not surface any foreground entry."""
        resp = seeded_client.get("/api/system/observability/token-usage?window=day&type=background")
        assert resp.status_code == 200
        body = json.loads(resp.data)

        types_in_response = {e["type"] for e in body["entries"]}
        assert types_in_response <= {"background"}, (
            f"type=background still surfaced non-background entries: {types_in_response}"
        )

    def test_no_filter_returns_both_types(self, seeded_client: FlaskClient) -> None:
        """Without ``type``, the response spans both types."""
        resp = seeded_client.get("/api/system/observability/token-usage?window=day")
        assert resp.status_code == 200
        body = json.loads(resp.data)

        types_in_response = {e["type"] for e in body["entries"]}
        assert types_in_response >= {"foreground", "background"}, (
            f"expected both types, got {types_in_response}"
        )

    # ------------------------------------------------------------------
    # Claim (d): summary totals mirror seeded numbers exactly.
    # ------------------------------------------------------------------

    def test_summary_totals_match_seeded_numbers_exact(self, seeded_client: FlaskClient) -> None:
        """Four query shapes (window x type-filter combinations) are verified
        against a hand-computed expectation derived solely from the seeded rows.

        Row breakdown of total tokens per row (input+output+cache_read+
        cache_create+thinking):
          foreground/x/alpha (recent):   100+20+10+2+5   =    137
          background/x/alpha (recent):    40+8+4+1+2     =     55
          foreground/y/beta  (recent):  200+40+20+4+10   =    274
          foreground/x/alpha (1yr ago): 1M+500K+100K+50K+25K = 1_675_000

        day_no_filter      = 137+55+274 = 466
        day_foreground     = 137+274 = 411
        day_background     = 55
        lifetime_no_filter = 466 + 1_675_000

        A bug merging foreground+background into one row would either leave
        day_background inflated or day_foreground reduced by the wrong amount;
        comparing both branches independently catches precisely that.
        """
        cases = [
            ("/api/system/observability/token-usage?window=day",
             "day_no_filter", _RECENT_TOTAL),
            ("/api/system/observability/token-usage?window=day&type=foreground",
             "day_foreground", _FOREGROUND_RECENT),
            ("/api/system/observability/token-usage?window=day&type=background",
             "day_background", _BACKGROUND_RECENT),
            ("/api/system/observability/token-usage?window=lifetime",
             "lifetime_no_filter", _LIFETIME_TOTAL),
        ]

        for url, label, expected_total in cases:
            resp = seeded_client.get(url)
            assert resp.status_code == 200, (
                f"{label}: HTTP {resp.status_code} — {resp.data[:300]}"
            )
            body = json.loads(resp.data)
            observed = int(body["summary"]["total_tokens"])
            assert observed == expected_total, (
                f"{label}: expected total_tokens={expected_total}, got {observed}; "
                f"entries: {body['entries']}"
            )

    # ------------------------------------------------------------------
    # Claim (e): window bound excludes old rows.
    # ------------------------------------------------------------------

    def test_window_bound_excludes_old_row(self, seeded_client: FlaskClient) -> None:
        """A row seeded a year ago (relative to SQLite's own clock) contributes
        ZERO to every window shorter than lifetime.

        We exercise all three finite windows (day, week, month) and verify
        that every reported bucket lies within the window *and* that the
        total matches the independent hand-computed sum of the recent rows.
        """
        for window in ("day", "week", "month"):
            resp = seeded_client.get(f"/api/system/observability/token-usage?window={window}")
            assert resp.status_code == 200, (
                f"?window={window}: HTTP {resp.status_code} — {resp.data[:300]}"
            )
            body = json.loads(resp.data)
            assert body["window"] == window

            observed_total = int(body["summary"]["total_tokens"])
            assert observed_total == _RECENT_TOTAL, (
                f"window={window}: expected {_RECENT_TOTAL} but got {observed_total}; "
                f"entries: {json.dumps(body['entries'], indent=2)}"
            )

            # Every returned bucket must lie within the window.
            # (Extra check on top of the numeric comparison, so a bucket drift
            #  bug would fail even if the totals accidentally coincided.)
            now = datetime.now(timezone.utc)
            max_diff_seconds = {
                "day": 24 * 3600,
                "week": 7 * 24 * 3600,
                "month": 31 * 24 * 3600,
            }[window]
            # Bucket format mirrors LlmLogService.token_usage exactly: by-hour
            # for `day`/`hour`, by-day otherwise (see _WINDOW_OFFSETS usage).
            bucket_format = "%Y-%m-%dT%H:%M:%S" if window == "day" else "%Y-%m-%d"
            for entry in body["entries"]:
                bucket_dt = datetime.strptime(entry["bucket"], bucket_format).replace(
                    tzinfo=timezone.utc,
                )
                diff_seconds = abs((now - bucket_dt).total_seconds())
                assert diff_seconds < max_diff_seconds, (
                    f"entry {entry['bucket']} lies {diff_seconds}s outside {window} window"
                )

    # ------------------------------------------------------------------
    # Claim (f): an unrecognised window is rejected cleanly, not crashed on.
    # ------------------------------------------------------------------

    def test_invalid_window_returns_422(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """An unknown ``window`` value returns 422 with a clean message —
        never a 500. Without the route's own membership check, an unrecognised
        window falls through to ``_WINDOW_OFFSETS[window]`` inside
        ``LlmLogService.token_usage`` and raises ``KeyError``, which the route
        turns into an opaque 500. That is the regression this test is proof
        against: a malformed query string from the Brain UI must fail loud and
        specific, not with a generic server error.
        """
        client = cast(FlaskClient, authed_client[0])
        resp = client.get("/api/system/observability/token-usage?window=biennium")
        assert resp.status_code == 422, (
            f"expected 422 for unknown window='biennium', got {resp.status_code}"
        )
