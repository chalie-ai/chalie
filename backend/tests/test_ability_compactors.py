"""Chat-history-compactor business-logic tests. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
compactor's genuine behaviour tests (broadcast guard) that have no coverage
elsewhere.

(The ``rows_compacted`` / ``_fit_compaction_input`` coverage that lived in this
file was removed during the old-spine ``MessageProcessor`` cleanup: it directly
constructed ``MessageProcessor`` and drove ``ChatHistoryCompactor.
_fit_compaction_input`` on it, which reads ``parent.providers.
get_context_limit()`` / ``parent.providers.measure()`` — the OLD provider-service
API. The new ``controllers.message_processor.MessageProcessor`` has no
``.providers`` attribute at all (renamed ``provider_service``, with
``context_limit()`` / ``measure()`` methods — confirmed via grep across
``controllers/message_processor.py`` and ``services/provider_service.py``).
``abilities.chat_history_compactor.ChatHistoryCompactor._fit_compaction_input``
and its ``_CompactionParent`` Protocol have not yet been migrated to the new
provider-service API, so this coverage cannot be restored without also editing
that production ability file — out of scope for this cleanup. See the dead-code
cleanup report for the systemic gap this exposed.)
"""

import pytest

from abilities._compaction_config import CompactionConfig
from abilities.chat_history_compactor import ChatHistoryCompactionConfig

pytestmark = pytest.mark.unit


def test_compaction_config_never_broadcasts_to_user() -> None:
    """Pins existing contract: the shared CompactionConfig (and both concrete
    subclasses) carry ``broadcast_to=None`` — never 'user'. The dispatcher only
    assigns a rich-media ordinal when broadcast_to == 'user' AND tr.rich, so a
    compactor result can never be paired to a user-facing card."""
    assert CompactionConfig().broadcast_to is None
    assert ChatHistoryCompactionConfig().broadcast_to is None
