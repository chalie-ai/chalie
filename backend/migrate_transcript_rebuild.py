#!/usr/bin/env python3
"""
One-time migration: rebuild transcript table with clean north-star-aligned data.

IMPORTANT: Stop Chalie before running this script.

What this does:
  1. Reads the last N transcript entries for the specified channel(s)
  2. Groups into (user → assistant) exchange pairs; skips background turns
     (proactive_thought, subagent, scheduled) and tool-result rows
  3. Strips the UserPromptAssemblyService wrapper from user content — keeps
     only the raw user message below the "## User Message" marker
  4. Deduplicates pairs by content hash
  5. Deletes orphaned tool_calls, clears compactions, truncates transcript
     rows for the affected channels
  6. Re-inserts clean (user, assistant) pairs in chronological order
     WITHOUT tool_calls rows — north-star format: no role='tool' in transcript,
     no tool_calls table entries (tool call intermediate work is intentionally lost)

Why tool calls are dropped:
  build_messages() only adds tool_calls[] to an assistant message if there are
  matching tool_calls table rows with tool_call_id IS NOT NULL.  With no
  re-created tool_calls rows, assistant messages rebuild with an empty
  tool_calls[] — valid provider format, no orphaned tool/result sequences.

Collateral tables:
  - episodes: transcript_id_start/end references in episodes become orphaned
    integer IDs. SQLite does not enforce this FK, and episode retrieval uses
    vector similarity, not transcript ID lookup — harmless.
  - memory_recall_log: same orphaned-ID situation, same non-impact.
  - compactions: watermark ID is invalidated; row is deleted and rebuilt
    naturally during normal operation.

Usage:
  # Dry run — show what would happen, change nothing
  python migrate_transcript_rebuild.py --dry-run

  # Run against default DB (backend/data/chalie.db), channel='user'
  python migrate_transcript_rebuild.py

  # Custom DB path
  python migrate_transcript_rebuild.py --db /data/chalie.db

  # Different channel or limit
  python migrate_transcript_rebuild.py --channel main --limit 200
"""

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── DB path resolution ──────────────────────────────────────────────────────

_DEFAULT_DB = str(Path(__file__).resolve().parent / "data" / "chalie.db")


def _resolve_db(cli_arg: str | None) -> str:
    if cli_arg:
        return cli_arg
    return os.environ.get("CHALIE_DB_PATH", _DEFAULT_DB)


# ── Content extraction ──────────────────────────────────────────────────────

# Marker written by UserPromptAssemblyService._build_current_turn()
_USER_MSG_MARKER = "## User Message\n"

# Assembly markers that indicate this is a wrapped prompt, not raw text
_ASSEMBLY_MARKERS = (
    "## World State",  # legacy (pre-0.3.3)
    "### Background Telemetry,Processes & Signals",  # post-0.3.3 WorldState header
    "# Current Turn",
    "### Related Memories",
    "## System Awareness",
    "IMPORTANT: The user is in voice mode",
)

# Content prefixes indicating a system/scheduled injection, not a user message
_INJECTION_PREFIXES = (
    "[SCHEDULED TASK",   # ScheduledMessageProcessor prompts
    "1. [20",            # Episode-dump injections (e.g. "1. [2026-04-08 06:40] salience:2 ...")
)

# Max chars for the plain-text (no-marker) fallback path — longer content is
# almost certainly a bulk injection or legacy context dump, not a user message
_PLAIN_TEXT_MAX = 10_000

# Roles that are background/internal — skip entirely
_SKIP_ROLES = {"proactive_thought", "subagent", "subagent_return", "scheduled", "tool"}


def extract_user_message(content: str) -> str | None:
    """Extract the raw user message from an assembled prompt.

    Returns the extracted text, or None if unrecoverable.

    Three cases:
      1. Has '## User Message\\n' marker: extract everything after it.
         If the extracted text itself contains assembly headers, the entry
         is a recursive blob (world-state injected into the prior turn's
         content, then re-assembled) — skip it.
      2. No assembly headers at all, short enough, no injection prefix:
         treat as raw user text (old format before UserPromptAssemblyService).
      3. Anything else: unrecoverable — skip.
    """
    if not content:
        return None

    # Case 1: Has the assembly marker — extract after it
    if _USER_MSG_MARKER in content:
        idx = content.index(_USER_MSG_MARKER) + len(_USER_MSG_MARKER)
        raw = content[idx:].strip()
        if len(raw) < 3:
            return None
        # Recursive blob check: after extraction the text still contains
        # assembly structure from the previous turn embedded inside it
        if any(m in raw for m in _ASSEMBLY_MARKERS):
            return None
        return raw

    # Case 2: No assembly markers — old plain-text format
    if not any(m in content for m in _ASSEMBLY_MARKERS):
        stripped = content.strip()
        if len(stripped) < 3:
            return None
        # Skip known system/scheduled injection patterns
        if any(stripped.startswith(p) for p in _INJECTION_PREFIXES):
            return None
        # Skip bulk dumps that are too long to be a real user message
        if len(stripped) > _PLAIN_TEXT_MAX:
            return None
        return stripped

    # Case 3: Has assembly headers but no ## User Message marker
    return None


def _pair_key(user_content: str, assistant_content: str) -> str:
    h = hashlib.sha256((user_content + "\x00" + assistant_content).encode()).hexdigest()
    return h[:24]


# ── Core migration ──────────────────────────────────────────────────────────

def run_migration(db_path: str, channel: str, limit: int, dry_run: bool):
    """Main migration routine for one channel.  Returns a result dict."""

    if not Path(db_path).exists():
        print(f"  ERROR: DB not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")  # we handle referential cleanup manually

    try:
        return _do_migration(conn, channel, limit, dry_run)
    finally:
        conn.close()


def _do_migration(conn: sqlite3.Connection, channel: str, limit: int, dry_run: bool) -> dict:
    stats = {
        "channel": channel,
        "rows_read": 0,
        "skipped_role": 0,
        "skipped_unrecoverable_user": 0,
        "skipped_empty_assistant": 0,
        "skipped_unpaired": 0,
        "skipped_dedup": 0,
        "pairs_inserted": 0,
        "orphaned_tool_calls_deleted": 0,
        "unrecoverable_details": [],
    }

    # ── 1. Read last N rows for this channel ──────────────────────────────

    rows = conn.execute(
        """
        SELECT id, role, content, tool_call_id, tool_name, created_at
        FROM transcript
        WHERE channel = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (channel, limit),
    ).fetchall()

    if not rows:
        print(f"  No transcript rows found for channel={channel!r}")
        return stats

    # Reverse to chronological order (oldest first)
    rows = list(reversed(rows))
    stats["rows_read"] = len(rows)

    # ── 2. Walk rows and build (user, assistant) exchange pairs ───────────
    #
    # Algorithm:
    #   - buffer the most-recent user row we've seen
    #   - when we hit an assistant row, emit a pair and clear the buffer
    #   - consecutive user rows: replace buffer (last user before assistant wins)
    #   - skip roles in _SKIP_ROLES

    pairs = []  # list of (user_row, assistant_row)
    pending_user = None

    for row in rows:
        role = row["role"]

        if role in _SKIP_ROLES:
            stats["skipped_role"] += 1
            continue

        if role == "user":
            pending_user = row
            continue

        if role == "assistant":
            if pending_user is None:
                # Assistant with no preceding user in this window — skip
                stats["skipped_unpaired"] += 1
                continue
            pairs.append((pending_user, row))
            pending_user = None
            continue

        # Any other role we don't recognise — skip silently
        stats["skipped_role"] += 1

    # A trailing user row with no following assistant — unpaired
    if pending_user is not None:
        stats["skipped_unpaired"] += 1

    # ── 3. Clean user content + dedup ────────────────────────────────────

    clean_pairs = []
    seen_keys: set[str] = set()

    for user_row, asst_row in pairs:
        # Extract raw user message
        raw_user = extract_user_message(user_row["content"])
        if raw_user is None:
            stats["skipped_unrecoverable_user"] += 1
            stats["unrecoverable_details"].append(
                f"  id={user_row['id']} created_at={user_row['created_at']} "
                f"  preview={user_row['content'][:80]!r}"
            )
            continue

        # Validate assistant content
        raw_asst = (asst_row["content"] or "").strip()
        if not raw_asst:
            stats["skipped_empty_assistant"] += 1
            continue

        # Dedup
        key = _pair_key(raw_user, raw_asst)
        if key in seen_keys:
            stats["skipped_dedup"] += 1
            continue
        seen_keys.add(key)

        clean_pairs.append({
            "user_content": raw_user,
            "user_created_at": user_row["created_at"],
            "asst_content": raw_asst,
            "asst_created_at": asst_row["created_at"],
        })

    # ── 4. Report what we found ───────────────────────────────────────────

    print(f"\n  Channel: {channel!r}")
    print(f"  Rows read            : {stats['rows_read']}")
    print(f"  Skipped (role)       : {stats['skipped_role']}  "
          f"(tool, proactive_thought, subagent, scheduled)")
    print(f"  Skipped (unpaired)   : {stats['skipped_unpaired']}  "
          f"(user with no following assistant, or assistant with no preceding user)")
    print(f"  Skipped (bad user)   : {stats['skipped_unrecoverable_user']}  "
          f"(cannot extract raw message from assembled prompt)")
    print(f"  Skipped (empty asst) : {stats['skipped_empty_assistant']}")
    print(f"  Skipped (dedup)      : {stats['skipped_dedup']}")
    print(f"  Clean pairs to write : {len(clean_pairs)}")

    if stats["unrecoverable_details"]:
        print("\n  Unrecoverable user rows (skipped):")
        for d in stats["unrecoverable_details"]:
            print(d)

    if not clean_pairs:
        print("\n  Nothing to write.")
        return stats

    if dry_run:
        print("\n  [DRY RUN] No changes made.")
        _print_pair_preview(clean_pairs[:3])
        return stats

    # ── 5. Delete old data ────────────────────────────────────────────────

    # Collect IDs of rows we're about to delete (for vec cleanup)
    old_ids = [r["id"] for r in rows]

    with conn:
        # 5a. Delete tool_calls rows linked to transcript rows for this channel
        if old_ids:
            placeholders = ",".join("?" * len(old_ids))
            cur = conn.execute(
                f"DELETE FROM tool_calls WHERE transcript_id IN ({placeholders})",
                old_ids,
            )
            stats["orphaned_tool_calls_deleted"] = cur.rowcount

        # 5b. Delete compaction (watermark is stale post-truncate)
        conn.execute("DELETE FROM compactions WHERE channel = ?", (channel,))

        # 5c. Delete transcript rows for this channel
        #     (we only delete the rows we READ — the last N — not the whole channel,
        #      since rows earlier than our window are already compacted / irrelevant)
        if old_ids:
            conn.execute(
                f"DELETE FROM transcript WHERE id IN ({placeholders})",
                old_ids,
            )

        # ── 6. Re-insert clean pairs ─────────────────────────────────────

        for pair in clean_pairs:
            conn.execute(
                """
                INSERT INTO transcript (channel, role, content, internal, created_at, xml_migrated)
                VALUES (?, 'user', ?, 0, ?, 1)
                """,
                (channel, pair["user_content"], pair["user_created_at"]),
            )
            conn.execute(
                """
                INSERT INTO transcript (channel, role, content, internal, created_at, xml_migrated)
                VALUES (?, 'assistant', ?, 0, ?, 1)
                """,
                (channel, pair["asst_content"], pair["asst_created_at"]),
            )

    stats["pairs_inserted"] = len(clean_pairs)
    print(f"\n  Written {len(clean_pairs)} exchange pairs "
          f"({len(clean_pairs) * 2} transcript rows)")
    print(f"  Deleted {stats['orphaned_tool_calls_deleted']} orphaned tool_calls rows")
    print("  Compaction cleared (will rebuild naturally)")
    print()
    print("  NOTE: build_messages() is unaffected — it reads by ID range.")

    return stats


def _print_pair_preview(pairs: list):
    print("\n  Preview (first 3 pairs):")
    for i, pair in enumerate(pairs):
        u_preview = pair["user_content"][:120].replace("\n", " ")
        a_preview = pair["asst_content"][:120].replace("\n", " ")
        print(f"    [{i + 1}] user     {pair['user_created_at']}  {u_preview!r}")
        print(f"         assistant {pair['asst_created_at']}  {a_preview!r}")
        print()


# ── Boot-time auto-run ──────────────────────────────────────────────────────

# Sentinel file written after the migration completes.  Its presence prevents
# re-running on subsequent boots.
_MIGRATION_ID = "transcript-rebuild-v1"

# Channel prefixes that belong to background/system flows — skip during auto-run.
# prompt:  — old scheduled-prompt delivery channels (contain [SCHEDULED TASK] text)
# web:     — old web interface channels (also contained scheduled prompts)
# scheduled: — new ScheduledMessageProcessor channels
_BACKGROUND_PREFIXES = ("subagent:", "scheduled:", "dmn:", "cron_tool:", "prompt:", "web:")


def _flag_path(db_path: str) -> Path:
    """Return the path of the done-flag file for this migration."""
    data_dir = Path(db_path).parent
    return data_dir / f".{_MIGRATION_ID}.done"


def run_once_on_boot(db_path: str | None = None, limit: int = 100) -> None:
    """Run the transcript rebuild migration exactly once at boot time.

    Checks for a sentinel file alongside the DB.  If absent, runs the
    migration silently on all non-background channels and writes the sentinel
    on success.  Subsequent boots are no-ops.

    Safe to call even if the DB has no transcript rows — the per-channel
    loop exits early with "Nothing to write."

    Args:
        db_path: Override the DB path (default: $CHALIE_DB_PATH or built-in default).
        limit:   Number of most-recent rows to analyse per channel (default 100).
    """
    db_path = _resolve_db(db_path)
    flag = _flag_path(db_path)

    if flag.exists():
        logger.debug(f"[TranscriptMigration] Already done (sentinel {flag})")
        return

    if not Path(db_path).exists():
        logger.warning(
            f"[TranscriptMigration] DB not found at {db_path} — skipping"
        )
        return

    logger.info("[TranscriptMigration] Running one-time transcript rebuild …")

    try:
        # Discover all channels that have transcript rows
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        all_channels = [
            r["channel"]
            for r in conn.execute(
                "SELECT DISTINCT channel FROM transcript ORDER BY channel"
            ).fetchall()
        ]
        conn.close()

        # Skip background/system channels
        channels = [
            ch for ch in all_channels
            if not any(ch.startswith(p) for p in _BACKGROUND_PREFIXES)
        ]

        if not channels:
            logger.info("[TranscriptMigration] No user-facing channels found — nothing to do")
            _write_flag(flag)
            return

        logger.info(f"[TranscriptMigration] Channels to migrate: {channels}")

        total_pairs = 0
        total_skipped = 0
        for ch in channels:
            result = run_migration(db_path, ch, limit, dry_run=False)
            total_pairs += result.get("pairs_inserted", 0)
            total_skipped += (
                result.get("skipped_unrecoverable_user", 0)
                + result.get("skipped_empty_assistant", 0)
                + result.get("skipped_unpaired", 0)
                + result.get("skipped_dedup", 0)
            )

        logger.info(
            f"[TranscriptMigration] Complete — "
            f"{total_pairs} pair(s) written, {total_skipped} skipped"
        )
        _write_flag(flag)

    except Exception as e:
        # Never crash boot — log and carry on
        logger.error(
            f"[TranscriptMigration] Failed: {e}",
            exc_info=True,
        )


def _write_flag(flag: Path) -> None:
    try:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(_MIGRATION_ID)
        logger.debug(f"[TranscriptMigration] Sentinel written: {flag}")
    except Exception as e:
        logger.warning(f"[TranscriptMigration] Could not write sentinel: {e}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rebuild transcript table with clean north-star-aligned exchange pairs."
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help=f"SQLite DB path (default: $CHALIE_DB_PATH or {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--channel",
        default="user",
        help="Channel to migrate (default: 'user'). Use 'ALL' to migrate every channel.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of most-recent transcript rows to analyse per channel (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and report only — make no changes to the database",
    )
    args = parser.parse_args()

    db_path = _resolve_db(args.db)
    print(f"DB path  : {db_path}")
    print(f"Mode     : {'DRY RUN — no changes will be made' if args.dry_run else 'LIVE — database will be modified'}")
    print(f"Limit    : last {args.limit} rows per channel")

    if not args.dry_run:
        print()
        confirm = input("Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # Resolve channels
    if args.channel.upper() == "ALL":
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        channels = [
            r["channel"]
            for r in conn.execute(
                "SELECT DISTINCT channel FROM transcript ORDER BY channel"
            ).fetchall()
        ]
        conn.close()
        print(f"\nChannels : {channels}")
    else:
        channels = [args.channel]

    total_pairs = 0
    for ch in channels:
        result = run_migration(db_path, ch, args.limit, args.dry_run)
        total_pairs += result.get("pairs_inserted", 0)

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("DRY RUN complete — database unchanged.")
    else:
        print(f"Migration complete. {total_pairs} clean exchange pair(s) written.")
        print()
        print("Next steps:")
        print("  1. Start Chalie normally — it will rebuild its context window from the")
        print("     clean transcript on the first user message.")
        print("  2. Compaction will re-run automatically when context reaches 80%.")
        print("  3. Episodic extraction fires on the per-channel rolling trigger (first at 25 inserts, then every 20); new exchanges will be embedded.")


if __name__ == "__main__":
    main()
