"""
Unit tests for ConsequenceClassifierService.

Tests cover:
  - Rule-based classification for all four tiers (clear keyword matches)
  - Edge cases: mixed keywords (higher tier wins), no keywords (default ACT),
    empty description, case-insensitivity
  - is_reversible() for all tiers
  - Output format: all required keys present with correct types
  - Graceful fallback when ONNX model is absent (rule-based takes over)
  - Singleton accessor returns same instance
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.consequence_classifier_service import (
    ConsequenceClassifierService,
    _rule_based_classify,
    get_consequence_classifier_service,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_svc() -> ConsequenceClassifierService:
    """Return a fresh service instance with no ONNX model path."""
    return ConsequenceClassifierService(models_dir="/nonexistent/models/path")


def _assert_result_schema(result: dict):
    """Assert that all required keys are present with the right types."""
    assert "tier" in result, "missing key: tier"
    assert "tier_name" in result, "missing key: tier_name"
    assert "confidence" in result, "missing key: confidence"
    assert "scores" in result, "missing key: scores"
    assert "method" in result, "missing key: method"

    assert isinstance(result["tier"], int), f"tier must be int, got {type(result['tier'])}"
    assert isinstance(result["tier_name"], str), "tier_name must be str"
    assert isinstance(result["confidence"], float), "confidence must be float"
    assert isinstance(result["scores"], dict), "scores must be dict"
    assert isinstance(result["method"], str), "method must be str"

    assert 0 <= result["tier"] <= 3, f"tier out of range: {result['tier']}"
    assert 0.0 <= result["confidence"] <= 1.0, f"confidence out of range: {result['confidence']}"

    # scores dict must contain all four tier names
    for key in ("observe", "organize", "act", "commit"):
        assert key in result["scores"], f"scores missing key: {key}"
        assert isinstance(result["scores"][key], float), f"scores[{key!r}] must be float"

    assert result["tier_name"] == ConsequenceClassifierService.TIER_NAMES[result["tier"]]
    assert result["method"] in ("onnx", "rule_based")


# ── Tier constants ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTierConstants:
    def test_tier_values(self):
        assert ConsequenceClassifierService.OBSERVE == 0
        assert ConsequenceClassifierService.ORGANIZE == 1
        assert ConsequenceClassifierService.ACT == 2
        assert ConsequenceClassifierService.COMMIT == 3

    def test_tier_names_mapping(self):
        names = ConsequenceClassifierService.TIER_NAMES
        assert names[0] == "observe"
        assert names[1] == "organize"
        assert names[2] == "act"
        assert names[3] == "commit"


# ── Rule-based: OBSERVE tier ──────────────────────────────────────────────────


@pytest.mark.unit
class TestRuleBasedObserve:
    def test_research(self):
        r = _rule_based_classify("research the history of the Roman Empire")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE
        assert r["tier_name"] == "observe"
        assert r["method"] == "rule_based"

    def test_search(self):
        r = _rule_based_classify("search for Python tutorials online")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_read(self):
        r = _rule_based_classify("read the article about machine learning")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_recall(self):
        r = _rule_based_classify("recall what the user said about their job")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_fetch(self):
        r = _rule_based_classify("fetch the weather forecast for tomorrow")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_introspect(self):
        r = _rule_based_classify("introspect current identity and memory state")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_analyse(self):
        r = _rule_based_classify("analyse the user's recent communication style")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_scores_sum(self):
        r = _rule_based_classify("look up the nearest coffee shop")
        assert r["scores"]["observe"] == 1.0
        assert r["scores"]["organize"] == 0.0
        assert r["scores"]["act"] == 0.0
        assert r["scores"]["commit"] == 0.0


# ── Rule-based: ORGANIZE tier ─────────────────────────────────────────────────


@pytest.mark.unit
class TestRuleBasedOrganize:
    def test_note(self):
        r = _rule_based_classify("create a note about the project requirements")
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE
        assert r["tier_name"] == "organize"

    def test_memorize(self):
        r = _rule_based_classify("memorize that the user prefers dark mode")
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE

    def test_list(self):
        r = _rule_based_classify("add the item to the grocery list")
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE

    def test_tag(self):
        r = _rule_based_classify("tag the document as high priority")
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE

    def test_categorize(self):
        r = _rule_based_classify("categorize the uploaded file under finance")
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE

    def test_associate(self):
        r = _rule_based_classify("associate this concept with machine learning")
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE

    def test_scores_sum(self):
        r = _rule_based_classify("organize my reading list by topic")
        assert r["scores"]["organize"] == 1.0
        assert r["scores"]["observe"] == 0.0
        assert r["scores"]["act"] == 0.0
        assert r["scores"]["commit"] == 0.0


# ── Rule-based: ACT tier ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestRuleBasedAct:
    def test_send(self):
        r = _rule_based_classify("send an email to the team about the meeting")
        assert r["tier"] == ConsequenceClassifierService.ACT
        assert r["tier_name"] == "act"

    def test_schedule(self):
        r = _rule_based_classify("schedule a reminder for next Monday")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_notify(self):
        r = _rule_based_classify("notify the user about the upcoming deadline")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_post(self):
        r = _rule_based_classify("post an update to the project channel")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_book(self):
        r = _rule_based_classify("book the conference room for Friday")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_reply(self):
        r = _rule_based_classify("reply to the message from Alice")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_scores_sum(self):
        r = _rule_based_classify("send a message to the manager")
        assert r["scores"]["act"] == 1.0
        assert r["scores"]["observe"] == 0.0
        assert r["scores"]["organize"] == 0.0
        assert r["scores"]["commit"] == 0.0


# ── Rule-based: COMMIT tier ───────────────────────────────────────────────────


@pytest.mark.unit
class TestRuleBasedCommit:
    def test_delete(self):
        r = _rule_based_classify("delete all the project files from the server")
        assert r["tier"] == ConsequenceClassifierService.COMMIT
        assert r["tier_name"] == "commit"

    def test_purchase(self):
        r = _rule_based_classify("purchase the annual subscription plan")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_pay(self):
        r = _rule_based_classify("pay the invoice for the software license")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_buy(self):
        r = _rule_based_classify("buy the replacement keyboard from Amazon")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_spend(self):
        r = _rule_based_classify("spend the remaining budget on advertising")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_transfer_money(self):
        r = _rule_based_classify("transfer money to the supplier account")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_destroy(self):
        r = _rule_based_classify("destroy the database backup permanently")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_scores_sum(self):
        r = _rule_based_classify("delete the file")
        assert r["scores"]["commit"] == 1.0
        assert r["scores"]["observe"] == 0.0
        assert r["scores"]["organize"] == 0.0
        assert r["scores"]["act"] == 0.0


# ── Edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRuleBasedEdgeCases:
    def test_no_keywords_defaults_to_act(self):
        """Description with no recognised keywords defaults to ACT (safe direction)."""
        r = _rule_based_classify("do something with the thing")
        assert r["tier"] == ConsequenceClassifierService.ACT
        assert r["confidence"] == 0.5  # lower confidence to signal uncertainty

    def test_empty_string_defaults_to_act(self):
        """Empty string should be handled gracefully."""
        svc = _make_svc()
        r = svc.classify("")
        assert r["tier"] == ConsequenceClassifierService.ACT
        assert r["method"] == "rule_based"

    def test_whitespace_only_defaults_to_act(self):
        svc = _make_svc()
        r = svc.classify("   ")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_case_insensitive(self):
        """Keywords must match regardless of case."""
        r = _rule_based_classify("SEARCH for news articles")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

        r = _rule_based_classify("DELETE the backup")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_mixed_keywords_higher_tier_wins(self):
        """When both COMMIT and OBSERVE keywords appear, COMMIT wins."""
        r = _rule_based_classify("search for files to delete permanently")
        assert r["tier"] == ConsequenceClassifierService.COMMIT

    def test_mixed_act_and_organize_act_wins(self):
        """When both ACT and ORGANIZE keywords appear, ACT wins."""
        r = _rule_based_classify("send the note to the team")
        assert r["tier"] == ConsequenceClassifierService.ACT

    def test_mixed_organize_and_observe_organize_wins(self):
        """When both ORGANIZE and OBSERVE keywords appear, ORGANIZE wins."""
        r = _rule_based_classify("search and save the results to a note")
        # "save" (ORGANIZE) and "search" (OBSERVE) both match;
        # ORGANIZE scan runs before OBSERVE, but COMMIT/ACT have priority.
        # The result depends on scan order: COMMIT, ACT, ORGANIZE, OBSERVE.
        # "save" is ORGANIZE, "search" is OBSERVE → ORGANIZE wins.
        assert r["tier"] == ConsequenceClassifierService.ORGANIZE

    def test_keyword_exact_word_match(self):
        """Keywords use word-boundary matching — 'search' matches as a standalone word."""
        r = _rule_based_classify("I will search for the document")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_no_false_positive_from_substring(self):
        """'call' inside 'recall' must NOT trigger the ACT tier."""
        r = _rule_based_classify("recall what the user said about the meeting")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE

    def test_short_description(self):
        """Single-word descriptions work."""
        r = _rule_based_classify("recall")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE


# ── is_reversible ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestIsReversible:
    def test_observe_is_reversible(self):
        svc = _make_svc()
        assert svc.is_reversible(ConsequenceClassifierService.OBSERVE) is True

    def test_organize_is_reversible(self):
        svc = _make_svc()
        assert svc.is_reversible(ConsequenceClassifierService.ORGANIZE) is True

    def test_act_is_reversible(self):
        svc = _make_svc()
        assert svc.is_reversible(ConsequenceClassifierService.ACT) is True

    def test_commit_is_not_reversible(self):
        svc = _make_svc()
        assert svc.is_reversible(ConsequenceClassifierService.COMMIT) is False

    def test_reversibility_boundary(self):
        """Tier 2 reversible, Tier 3 not — boundary is clean."""
        svc = _make_svc()
        assert svc.is_reversible(2) is True
        assert svc.is_reversible(3) is False


# ── Output format ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestOutputFormat:
    def test_observe_schema(self):
        svc = _make_svc()
        r = svc.classify("research machine learning papers")
        _assert_result_schema(r)

    def test_organize_schema(self):
        svc = _make_svc()
        r = svc.classify("add item to my reading list")
        _assert_result_schema(r)

    def test_act_schema(self):
        svc = _make_svc()
        r = svc.classify("send an email to the client")
        _assert_result_schema(r)

    def test_commit_schema(self):
        svc = _make_svc()
        r = svc.classify("delete all temporary files")
        _assert_result_schema(r)

    def test_no_keyword_schema(self):
        svc = _make_svc()
        r = svc.classify("do the thing")
        _assert_result_schema(r)

    def test_method_is_rule_based_without_onnx(self):
        """Without ONNX model, method must always be 'rule_based'."""
        svc = _make_svc()
        for desc in [
            "search for information",
            "save a note",
            "send a message",
            "delete the record",
            "do something unrecognised",
        ]:
            r = svc.classify(desc)
            assert r["method"] == "rule_based", f"expected rule_based for {desc!r}"


# ── ONNX graceful fallback ────────────────────────────────────────────────────


@pytest.mark.unit
class TestOnnxFallback:
    def test_missing_model_falls_back_to_rule_based(self):
        """Service must work correctly when the ONNX model does not exist."""
        svc = ConsequenceClassifierService(models_dir="/nonexistent/path/to/models")
        r = svc.classify("search for recipes online")
        # Falls back to rule-based — should still classify OBSERVE
        assert r["tier"] == ConsequenceClassifierService.OBSERVE
        assert r["method"] == "rule_based"

    def test_missing_model_classify_commit(self):
        svc = ConsequenceClassifierService(models_dir="/nonexistent/path/to/models")
        r = svc.classify("delete the old backup files")
        assert r["tier"] == ConsequenceClassifierService.COMMIT
        assert r["method"] == "rule_based"

    def test_missing_model_full_schema(self):
        svc = ConsequenceClassifierService(models_dir="/nonexistent/path/to/models")
        r = svc.classify("schedule a meeting for tomorrow")
        _assert_result_schema(r)


# ── Singleton ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSingleton:
    def test_singleton_returns_same_instance(self):
        """get_consequence_classifier_service() must return the same object."""
        a = get_consequence_classifier_service()
        b = get_consequence_classifier_service()
        assert a is b

    def test_singleton_is_correct_type(self):
        svc = get_consequence_classifier_service()
        assert isinstance(svc, ConsequenceClassifierService)

    def test_singleton_classifies(self):
        """Singleton must classify correctly end-to-end."""
        svc = get_consequence_classifier_service()
        r = svc.classify("read the news")
        assert r["tier"] == ConsequenceClassifierService.OBSERVE
        _assert_result_schema(r)
