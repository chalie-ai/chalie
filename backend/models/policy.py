"""Policy — one ``policy`` row: a (channel, permission) → setting triple.

Active-record row-model (Rule 5 / §4.1). ``id`` is the DDL's own
``INTEGER PRIMARY KEY AUTOINCREMENT``, so the base's ``save``/``get``/``delete``
id-centric verbs apply unmodified. This model is the SOLE home of
``policy`` + ``policy_blocked_log`` row SQL; :class:`~services.policy_manager.PolicyManager`
reads and writes exclusively through it. Holds no mp, calls no service
(Rule-3 depth).

The ``(channel, permission)`` pair is the natural identifier — there is no
``policy_id`` foreign key on ``policy_blocked_log``, so the link is derived
from ``action_id == permission AND context == channel``. The read/write verbs
are bespoke ``@classmethod``\\ s keyed on that pair rather than the base's
id-centric ``get``/``save``/``delete``, mirroring how :class:`~models.setting.Setting`
keys its verbs on ``key``.

The interactive gate needs a concurrency-safe default-to-'ask' on a miss, so
:class:`~models.policy.Policy` owns a bespoke ``get_or_create_default`` that
wraps the check-then-write in ``Database.transaction()`` — the same atomic
block the dissolved service used."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from models.model import Model
from services.database import Database

if TYPE_CHECKING:
    from models.policy_blocked_log import PolicyBlockedLog


class Policy(Model):
    """One ``policy`` row, keyed by ``(channel, permission)``."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "channel", "permission", "setting",
    )

    @classmethod
    def get_table(cls) -> str:
        return "policy"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    channel: str
    permission: str
    setting: str

    # ── Reads ────────────────────────────────────────────────────────────────

    @classmethod
    def get_or_create_default(cls, channel: str, permission: str) -> str:
        """Return the current setting for ``(channel, permission)``, creating
        an ``'ask'`` default row on a miss — the concurrency-safe seed used
        by the interactive gate.

        Mirrors the dissolved ``PolicyManager._setting``: a check-then-write
        inside ``Database.transaction()`` so a race between two concurrent
        gates for the same ``(channel, permission)`` never inserts two rows."""
        with Database.transaction() as conn:
            row = conn.execute(
                "SELECT setting FROM policy WHERE channel = ? AND permission = ?",
                (channel, permission),
            ).fetchone()
            if row is not None:
                return str(row[0])
            conn.execute(
                "INSERT OR IGNORE INTO policy (channel, permission, setting) "
                "VALUES (?, ?, 'ask')",
                (channel, permission),
            )
            return "ask"

    @classmethod
    def upsert(cls, channel: str, permission: str, setting: str) -> int:
        """Single-cell upsert via ``ON CONFLICT``. Returns rows affected
        (0 on invalid input — the caller validates before calling)."""
        with Database.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO policy (channel, permission, setting) VALUES (?, ?, ?) "
                "ON CONFLICT(channel, permission) DO UPDATE SET setting = ?",
                (channel, permission, setting, setting),
            )
            return cur.rowcount

    @classmethod
    def insert_or_ignore(cls, channel: str, permission: str, setting: str) -> int:
        """``INSERT OR IGNORE`` for one row. Returns the rowcount (0 when the
        ``(channel, permission)`` pair already exists). Used by the seed loader."""
        with Database.transaction() as conn:
            return conn.execute(
                "INSERT OR IGNORE INTO policy (channel, permission, setting) VALUES (?, ?, ?)",
                (channel, permission, setting),
            ).rowcount

    @classmethod
    def clear_all(cls) -> int:
        """Wipe every row. Returns the rowcount."""
        with Database.transaction() as conn:
            return conn.execute("DELETE FROM policy").rowcount

    # ── Relationship ─────────────────────────────────────────────────────────

    def get_blocked_entry(self) -> list[PolicyBlockedLog]:
        """Every ``policy_blocked_log`` row derived from this policy's
        ``(channel, permission)`` — ordered most-recent first.

        The link is derived (no FK): ``action_id == permission AND context ==
        channel``. Pure structured filter — no raw SQL."""
        # Import here to avoid circular dependency at module load time
        from models.policy_blocked_log import PolicyBlockedLog  # noqa: PLC0415
        return (
            PolicyBlockedLog.filter("action_id", self.permission)
            .filter("context", self.channel)
            .order_by("created_at DESC")
            .get()
        )
