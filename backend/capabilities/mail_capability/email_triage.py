"""email_triage — embedding-based email classification.

Classifies email header dicts as noise, informational, or actionable using
cosine similarity against canonical intent anchors.  ``has_unsubscribe`` is
the only deterministic fast-path; everything else goes through the embeddings.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anchor text for each triage category
# ---------------------------------------------------------------------------

_TRIAGE_ANCHORS = {
    "noise": (
        "automated notification, marketing newsletter, system alert, "
        "promotional offer, subscription digest, no-reply bulk email"
    ),
    "actionable": (
        "action required, please reply, deadline approaching, invoice "
        "payment due, approval needed, confirm attendance, follow-up request"
    ),
    "informational": (
        "general information, status update, FYI announcement, meeting "
        "notes shared, project update summary, weekly report"
    ),
}

_anchor_cache: dict | None = None


def _get_anchor_embeddings() -> dict:
    """Return cached anchor embeddings, computing on first call."""
    global _anchor_cache
    if _anchor_cache is not None:
        return _anchor_cache
    from services.embedding_service import get_embedding_service
    svc = get_embedding_service()
    _anchor_cache = {
        cat: svc.generate_embedding_np(text)
        for cat, text in _TRIAGE_ANCHORS.items()
    }
    return _anchor_cache


def _build_email_text(item: dict) -> str:
    """Build classification text from email signals."""
    parts = []
    subj = item.get("subject", "")
    if subj:
        parts.append(subj)
    sender = item.get("from_name") or item.get("from_addr", "")
    if sender:
        parts.append(f"from {sender}")
    return " ".join(parts) if parts else "email message"


def classify_email(item: dict) -> str:
    """Classify an email header dict as noise, informational, or actionable.

    Uses embedding similarity against canonical intent anchors.
    Only ``has_unsubscribe`` is a deterministic fast path; threaded replies
    (``in_reply_to``) go through the embedding classifier because many
    automated systems (newsletters, GitHub notifications) set In-Reply-To.
    """
    if item.get("has_unsubscribe"):
        return "noise"

    from services.embedding_service import get_embedding_service
    svc = get_embedding_service()
    email_emb = svc.generate_embedding_np(_build_email_text(item))
    anchors = _get_anchor_embeddings()

    best_cat = "informational"
    best_sim = -1.0
    for cat, anchor_emb in anchors.items():
        sim = float(np.dot(email_emb, anchor_emb))
        if sim > best_sim:
            best_sim = sim
            best_cat = cat
    return best_cat
