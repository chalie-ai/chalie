"""Tests for capabilities/mail_capability/email_triage.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capabilities.mail_capability.email_triage import (
    _TRIAGE_ANCHORS,
    _build_email_text,
    classify_email,
)


# ---------------------------------------------------------------------------
# _build_email_text
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildEmailText:

    def test_subject_and_sender_name(self):
        item = {"subject": "Invoice due", "from_name": "ACME Corp", "from_addr": "billing@acme.com"}
        result = _build_email_text(item)
        assert "Invoice due" in result
        assert "ACME Corp" in result

    def test_falls_back_to_from_addr_when_no_name(self):
        item = {"subject": "Hello", "from_addr": "sender@example.com"}
        result = _build_email_text(item)
        assert "sender@example.com" in result

    def test_empty_item_returns_fallback(self):
        assert _build_email_text({}) == "email message"

    def test_missing_subject_uses_sender(self):
        item = {"from_name": "Bob", "from_addr": "bob@example.com"}
        result = _build_email_text(item)
        assert "Bob" in result
        assert "Invoice" not in result


# ---------------------------------------------------------------------------
# classify_email — has_unsubscribe fast path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyEmailFastPath:

    def test_has_unsubscribe_returns_noise(self):
        item = {
            "subject": "Weekly digest",
            "from_addr": "newsletter@example.com",
            "has_unsubscribe": True,
        }
        assert classify_email(item) == "noise"

    def test_has_unsubscribe_false_does_not_short_circuit(self):
        """has_unsubscribe=False must not short-circuit to noise."""
        item = {
            "subject": "Action required: approve PR",
            "from_addr": "bot@github.com",
            "has_unsubscribe": False,
        }
        mock_svc = MagicMock()
        # Return a vector that scores highest against "actionable" anchor
        actionable_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_svc.generate_embedding_np.return_value = actionable_vec

        anchor_vecs = {
            "noise": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "actionable": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "informational": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

        # get_embedding_service is imported lazily inside the function, so patch
        # at the source module and also bypass the anchor cache.
        with patch(
            "services.embedding_service.get_embedding_service",
            return_value=mock_svc,
        ), patch(
            "capabilities.mail_capability.email_triage._get_anchor_embeddings",
            return_value=anchor_vecs,
        ):
            result = classify_email(item)
        assert result == "actionable"


# ---------------------------------------------------------------------------
# classify_email — embedding-based classification
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyEmailEmbedding:

    def _classify_with_mock(self, item: dict, email_vec: np.ndarray, anchor_vecs: dict) -> str:
        mock_svc = MagicMock()
        mock_svc.generate_embedding_np.return_value = email_vec

        # get_embedding_service is a lazy local import — patch at the source.
        with patch(
            "services.embedding_service.get_embedding_service",
            return_value=mock_svc,
        ), patch(
            "capabilities.mail_capability.email_triage._get_anchor_embeddings",
            return_value=anchor_vecs,
        ):
            return classify_email(item)

    def test_closest_to_noise_anchor(self):
        email_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        anchors = {
            "noise": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "actionable": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "informational": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        result = self._classify_with_mock({"subject": "Promo"}, email_vec, anchors)
        assert result == "noise"

    def test_closest_to_actionable_anchor(self):
        email_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        anchors = {
            "noise": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "actionable": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "informational": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        result = self._classify_with_mock({"subject": "Approve invoice"}, email_vec, anchors)
        assert result == "actionable"


# ---------------------------------------------------------------------------
# _TRIAGE_ANCHORS structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTriageAnchors:

    def test_has_all_three_categories(self):
        assert set(_TRIAGE_ANCHORS.keys()) == {"noise", "actionable", "informational"}


