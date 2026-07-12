"""Shared JSON wire-projection for every data-model — persisted rows and
transient WS frames alike (Rule 5 / Rule 8).

A model exposes itself "as a dict and as a json string" (§1.2 Rule 5). The
dict projection genuinely differs by kind — a persisted ``Model`` filters to
its schema columns (``__dict__`` ∩ ``__columns__``), a transient
``WsMessage`` projects every public field — so ``to_dict`` stays abstract
here and is defined per subclass. The JSON step, however, is identical for
both: ``json.dumps(self.to_dict(), …)`` with one aware-``datetime`` →
ISO-8601 encoder that fails loudly on anything else. So it lives here ONCE —
a single wire-encoding source of truth, so a persisted-model frame and a
transient frame on the same socket can never encode a datetime two different
ways (Essential 8: ZERO code duplication).

Holds no ``mp``, imports no service, opens no connection (Rule-3 depth).
"""

from __future__ import annotations

import json
from datetime import datetime


class Serializable:
    """JSON wire-projection base: ``to_json`` over a per-subclass ``to_dict``."""

    def to_dict(self) -> dict[str, object]:
        """Project the instance to a plain dict. Defined per subclass — column-
        filtered on a persisted ``Model``, all-public-fields on a ``WsMessage``."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Project via :meth:`to_dict` to a JSON string."""
        return json.dumps(self.to_dict(), default=self._json_default)

    def _json_default(self, value: object) -> str:
        """Encode the one non-JSON-native type a field legitimately holds — an
        aware ``datetime`` — as ISO-8601. Anything else means a service attached
        a live object / raw payload to a field: fail loudly (re-raise ``json``'s
        native ``TypeError``) rather than ship a garbage ``repr`` onto the wire."""
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"{type(value).__name__} is not JSON-serializable")
