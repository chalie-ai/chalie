"""PolicyBlockedLog — one ``policy_blocked_log`` row: a record of a gated
action that was blocked.

Active-record row-model (Rule 5 / §4.1). ``id`` is the DDL's own
``INTEGER PRIMARY KEY AUTOINCREMENT``, so the base's ``save``/``get``/``delete``
id-centric verbs apply unmodified. This model is the SOLE home of
``policy_blocked_log`` row SQL; :class:`~services.policy_manager.PolicyManager`
reads and writes exclusively through it. Holds no mp, calls no service
(Rule-3 depth).

There is no ``policy_id`` foreign key — the link to the parent ``Policy`` is
derived from ``action_id == permission AND context == channel``. The write
verb is bespoke (keyed on the derived fields) rather than the base's id-centric
``save``, mirroring how :class:`~models.setting.Setting` keys its verbs on
``key``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from models.model import Model
from services.database import Database

if TYPE_CHECKING:
    from models.policy import Policy


class PolicyBlockedLog(Model):
    """One ``policy_blocked_log`` row: a blocked action record."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "action_id", "context", "reason", "params_json", "created_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "policy_blocked_log"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    action_id: str
    context: str
    reason: str
    params_json: str | None
    created_at: str

    # ── Writes ───────────────────────────────────────────────────────────────

    @classmethod
    def log_blocked(cls, action_id: str, context: str, reason: str, created_at: str) -> None:
        """Insert a blocked-action record. The caller supplies ``created_at``
        so the timestamp is consistent with the transaction boundary."""
        with Database.transaction():
            cls(
                action_id=action_id,
                context=context,
                reason=reason,
                created_at=created_at,
            ).save()

    @classmethod
    def clear_all(cls) -> int:
        """Wipe every row. Returns the rowcount."""
        with Database.transaction() as conn:
            return conn.execute("DELETE FROM policy_blocked_log").rowcount

    # ── Relationship ─────────────────────────────────────────────────────────

    def policy(self) -> Policy | None:
        """The parent ``Policy`` derived from ``(action_id, context)``.

        The link is derived (no FK): ``permission == action_id AND channel ==
        context``. Pure structured filter — no raw SQL."""
        # Import here to avoid circular dependency at module load time
        from models.policy import Policy  # noqa: PLC0415
        return (
            Policy.filter("permission", self.action_id)
            .filter("channel", self.context)
            .first()
        )
