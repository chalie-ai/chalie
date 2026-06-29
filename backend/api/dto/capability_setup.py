"""DTO for the capability setup body — a deliberately open, polymorphic credential bag.

The setup body is genuinely per-capability (mail wants ``email``/``password``/
``imap_host``; calendar wants different keys; etc.), so there is no fixed
cross-capability schema. This is the **one** DTO that relaxes the foundation
``extra="forbid"`` to ``extra="allow"``: arbitrary credential keys pass through
to ``cap.configure()``, which validates per-capability and raises ``ValueError``
(surfaced as 422). Credentials are write-only — they appear ONLY on this inbound
DTO and are NEVER echoed on any response DTO.
"""

from __future__ import annotations

from pydantic import ConfigDict

from .base import DTO


class CapabilitySetup(DTO):
    """Open credential body for ``POST /api/capabilities/<id>/setup``.

    No declared fields: the per-capability contract is documented on the route
    via ``@ns.doc``. ``extra="allow"`` lets arbitrary credential keys through.
    """

    model_config = ConfigDict(extra="allow")