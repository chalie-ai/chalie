"""
Integration tests for the mode-gate ONNX model.

Validates that the retrained model correctly routes:
  - Clearly conversational messages → RESPOND (sigmoid ≥ 0.85)
  - Messages with complete actionable intent → ACT (sigmoid < 0.85)

Key principle: The gate sees only the raw message — no conversation context.
If a message doesn't contain enough information to act on (missing subject,
vague pronoun, incomplete request), the LLM in RESPOND mode can ask for
clarification. That's conversational, not actionable.

ACT is for messages where there IS enough info to invoke a skill or tool:
  - "remind me to call mom at 5pm" → ACT (has what + when)
  - "remind me at 5pm" → RESPOND (remind about what? needs clarification)

These tests run against the actual ONNX model — not mocked.
"""

import logging
import pytest

logger = logging.getLogger(__name__)

# Requires ONNX model files on disk + onnxruntime + transformers
pytestmark = [pytest.mark.integration]


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def onnx_svc():
    """Load the real ONNX inference service with mode-gate model."""
    from services.onnx_inference_service import get_onnx_inference_service
    svc = get_onnx_inference_service()
    if not svc.is_available("mode-gate"):
        pytest.skip("mode-gate model not available")
    return svc


def _predict(onnx_svc, text: str) -> tuple:
    """Run mode-gate prediction, return (route, confidence).

    route='respond' if RESPOND label fires (sigmoid ≥ threshold from meta).
    route='act' otherwise.
    """
    # Training format: "{message}\nGate:" — must match at inference.
    gate_text = f"{text}\nGate:"
    predictions = onnx_svc.predict_multi_label("mode-gate", gate_text)
    respond_hit = next(
        ((lbl, c) for lbl, c in predictions if lbl.upper() == 'RESPOND'),
        None,
    )
    if respond_hit:
        return 'respond', respond_hit[1]
    return 'act', 0.0


# ── Helpers ───────────────────────────────────────────────────

def assert_respond(onnx_svc, text: str, min_confidence: float = 0.85):
    route, conf = _predict(onnx_svc, text)
    assert route == 'respond', (
        f"Expected RESPOND for {text!r}, got ACT (conf={conf:.3f})"
    )
    assert conf >= min_confidence, (
        f"RESPOND confidence too low for {text!r}: {conf:.3f} < {min_confidence}"
    )


def assert_act(onnx_svc, text: str):
    route, conf = _predict(onnx_svc, text)
    assert route == 'act', (
        f"Expected ACT for {text!r}, got RESPOND (conf={conf:.3f})"
    )


# ══════════════════════════════════════════════════════════════
# RESPOND — clearly conversational, no tools needed
# ══════════════════════════════════════════════════════════════

class TestRespondGreetings:
    """Simple greetings — pure conversation, zero tool need."""

    @pytest.mark.parametrize("msg", [
        "hello",
        "hi there",
        "hey",
        "good morning",
        "good evening",
        "howdy",
        "hi, how's it going?",
        "hey, long time no chat",
    ])
    def test_greetings(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondGratitude:
    """Thank you / appreciation — conversational feedback."""

    @pytest.mark.parametrize("msg", [
        "thanks",
        "thank you",
        "thank you so much",
        "appreciate it",
        "thanks a lot",
        "cheers, that was helpful",
        "thanks for explaining that",
    ])
    def test_gratitude(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondKnowledge:
    """Factual questions answerable from training knowledge."""

    @pytest.mark.parametrize("msg", [
        "what is photosynthesis",
        "explain quantum computing in simple terms",
        "what's the difference between TCP and UDP",
        "how does a neural network work",
        "what is the capital of France",
        "who wrote Romeo and Juliet",
        "what causes earthquakes",
        "explain the theory of relativity",
        "what is an API",
        "how do computers store data",
    ])
    def test_knowledge_questions(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondOpinionsAdvice:
    """Opinions and advice requests — conversational reasoning."""

    @pytest.mark.parametrize("msg", [
        "what do you think about electric cars",
        "should I learn Python or JavaScript first",
        "is it worth getting a master's degree",
        "what's your take on remote work",
        "do you think AI will replace programmers",
        "which is better, cats or dogs",
    ])
    def test_opinions(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondMathDefinitions:
    """Math and definitions — answerable from training data."""

    @pytest.mark.parametrize("msg", [
        "what's 15% of 230",
        "define ontology",
        "what does ephemeral mean",
        "what's the square root of 144",
        "convert 100 fahrenheit to celsius",
    ])
    def test_math_definitions(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondConversational:
    """Conversational continuations — no action implied."""

    @pytest.mark.parametrize("msg", [
        "that's interesting",
        "tell me more",
        "I agree with that",
        "that makes sense",
        "I see what you mean",
        "haha that's funny",
        "wow I didn't know that",
        "that's a good point",
        "I disagree actually",
        "can you elaborate on that",
        "why is that the case",
        "how so",
    ])
    def test_conversational(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondFeedback:
    """User feedback on Chalie's responses — no action needed."""

    @pytest.mark.parametrize("msg", [
        "that was helpful",
        "not what I meant",
        "perfect",
        "exactly what I was looking for",
        "you misunderstood me",
        "no that's wrong",
        "great explanation",
        "I meant something different",
    ])
    def test_feedback(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondMetaCapability:
    """Questions ABOUT capabilities — not requests to USE them."""

    @pytest.mark.parametrize("msg", [
        "what can you do",
        "what tools do you have",
        "can you search the web",
        "how do I set a reminder",
        "tell me about your scheduling feature",
        "what skills do you have",
        "how does your memory work",
        "what are you capable of",
    ])
    def test_meta_capability(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondEmotional:
    """Emotional sharing — pure conversation."""

    @pytest.mark.parametrize("msg", [
        "I'm feeling stressed today",
        "I had a really bad day",
        "I'm so excited about this project",
        "feeling a bit down",
        "I'm happy with how things turned out",
    ])
    def test_emotional(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondStorytelling:
    """User sharing stories or anecdotes."""

    @pytest.mark.parametrize("msg", [
        "so yesterday I was at the store and the funniest thing happened",
        "I just finished reading a great book",
        "my dog did the silliest thing today",
        "I've been thinking about what you said earlier",
    ])
    def test_storytelling(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondAboutFeatures:
    """Asking ABOUT features vs requesting them — subtle but RESPOND."""

    @pytest.mark.parametrize("msg", [
        "how does the reminder system work",
        "what's the difference between a task and a reminder",
        "explain how your memory works",
        "tell me about your list feature",
        "how do focus sessions help",
        "what happens when I set a reminder",
    ])
    def test_about_features(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


class TestRespondHypotheticals:
    """Hypothetical or educational questions — no action needed."""

    @pytest.mark.parametrize("msg", [
        "what would happen if the sun disappeared",
        "how would you explain gravity to a five year old",
        "if you could travel anywhere where would you go",
        "what's the best way to learn a new language",
        "how do people usually handle stress",
    ])
    def test_hypotheticals(self, onnx_svc, msg):
        assert_respond(onnx_svc, msg)


# ══════════════════════════════════════════════════════════════
# ACT — messages with COMPLETE actionable intent
#
# These contain enough information for a skill/tool to execute.
# The gate has no conversation context — only the raw message.
# If the message is missing key info (what, when, where), the
# LLM in RESPOND mode asks for clarification. That's not ACT.
# ══════════════════════════════════════════════════════════════

class TestActUrls:
    """Messages containing URLs — always need the read skill."""

    @pytest.mark.parametrize("msg", [
        "read this: https://example.com/article",
        "what does this page say https://news.ycombinator.com",
        "can you summarize https://en.wikipedia.org/wiki/Python",
        "check out this link https://github.com/some/repo",
        "https://example.com",
    ])
    def test_urls(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActSchedulingComplete:
    """Scheduling requests with complete info (what + when)."""

    @pytest.mark.parametrize("msg", [
        "remind me to call mom at 5pm",
        "schedule a meeting for next Tuesday",
        "remind me about the dentist appointment",
        "set an alarm for 7am",
        "wake me up in 30 minutes",
        "remind me to take my medication at 9pm",
    ])
    def test_scheduling(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActListManagementComplete:
    """List operations with specific items or list names."""

    @pytest.mark.parametrize("msg", [
        "add milk to my shopping list",
        "what's on my todo list",
        "remove eggs from the grocery list",
        "create a new list called weekend plans",
        "show me my lists",
        "add eggs, butter, and flour to the shopping list",
    ])
    def test_list_management(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActWebSearch:
    """Web search / current information — need external tools."""

    @pytest.mark.parametrize("msg", [
        "search for Python tutorials",
        "what's the latest news about AI",
        "look up the weather in London",
        "find me recipes for pasta",
        "what's the current Bitcoin price",
        "search for flights to Tokyo",
        "google the nearest coffee shop",
        "find reviews for the new iPhone",
    ])
    def test_web_search(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActDocuments:
    """Document search/management — need the document skill."""

    @pytest.mark.parametrize("msg", [
        "what does my warranty say about returns",
        "search my documents for the contract",
        "find that PDF I uploaded yesterday",
        "what's in my notes about the meeting",
    ])
    def test_documents(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActTaskCreationComplete:
    """Background task creation with specific topics."""

    @pytest.mark.parametrize("msg", [
        "research quantum computing over the next week",
        "start a background task to monitor the server",
    ])
    def test_task_creation(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActFocusSessions:
    """Focus session management — need the focus skill."""

    @pytest.mark.parametrize("msg", [
        "start a focus session",
        "am I focused right now",
        "end my focus session",
        "I want to focus for the next hour",
        "start a deep work session",
    ])
    def test_focus_sessions(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActMemoryOperationsComplete:
    """Explicit memory operations with specific content."""

    @pytest.mark.parametrize("msg", [
        "remember that my favorite color is blue",
        "remember my wife's birthday is March 15th",
        "memorize this phone number: 555-1234",
        "what do you remember about our last conversation",
        "do you recall what I said about the project",
    ])
    def test_memory_operations(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


class TestActRealTimeData:
    """Questions needing real-time / current data."""

    @pytest.mark.parametrize("msg", [
        "what time is it",
        "what's today's date",
        "what day is it",
        "how's the weather",
        "what's the stock market doing",
        "is it raining outside",
    ])
    def test_real_time_data(self, onnx_svc, msg):
        assert_act(onnx_svc, msg)


# ══════════════════════════════════════════════════════════════
# AGGREGATE QUALITY — overall model behavior
# ══════════════════════════════════════════════════════════════

class TestAggregateQuality:
    """Aggregate metrics to catch systematic bias.

    Tests only clear-cut cases — messages that are unambiguously RESPOND
    or unambiguously ACT. Borderline cases (incomplete requests, vague
    pronouns, context-dependent messages) are intentionally excluded.
    """

    RESPOND_CASES = [
        # Greetings
        "hello", "hi there", "good morning",
        # Gratitude
        "thanks", "thank you so much",
        # Knowledge
        "what is photosynthesis", "explain neural networks",
        "what's the capital of Japan",
        # Opinions
        "should I learn Python or Go",
        "what do you think about climate change",
        # Conversational
        "that's interesting", "tell me more", "I agree",
        # Feedback
        "perfect", "great explanation", "not what I meant",
        # Meta
        "what can you do", "how does your memory work",
        # Emotional
        "I'm feeling stressed", "I had a great day",
        # Math
        "what's 12 times 15",
    ]

    ACT_CASES = [
        # URLs (always actionable — the URL IS the content)
        "read this https://example.com",
        # Complete scheduling (has what + when)
        "remind me to call mom at 5pm",
        "remind me about the dentist appointment",
        # Complete list ops (has item + list)
        "add milk to my shopping list", "show me my todo list",
        # Search (has query)
        "search for Python news", "look up the weather in London",
        # Documents (has search target)
        "find that contract", "search my documents for the lease",
        # Complete memory ops (has content)
        "remember my birthday is June 5th",
        # Real-time data
        "what time is it", "what's today's date",
        # Focus sessions (specific action)
        "start a focus session",
    ]

    def test_respond_precision(self, onnx_svc):
        """RESPOND precision must be ≥ 0.95.

        Of all messages the model routes to RESPOND, at least 95% must
        actually be conversational. This is the critical metric — false
        positives (action messages routed to RESPOND) are the dangerous
        failure mode.
        """
        false_positives = []
        for msg in self.ACT_CASES:
            route, conf = _predict(onnx_svc, msg)
            if route == 'respond':
                false_positives.append((msg, conf))

        true_positives = 0
        for msg in self.RESPOND_CASES:
            route, conf = _predict(onnx_svc, msg)
            if route == 'respond':
                true_positives += 1

        total_respond = true_positives + len(false_positives)
        if total_respond == 0:
            pytest.skip("Model routes everything to ACT — precision undefined")

        precision = true_positives / total_respond
        logger.info(
            f"RESPOND precision: {precision:.3f} "
            f"(TP={true_positives}, FP={len(false_positives)}, "
            f"total_respond={total_respond})"
        )

        if false_positives:
            fp_list = "\n".join(f"  - {msg!r} (conf={c:.3f})" for msg, c in false_positives)
            logger.warning(f"False positives (ACT messages routed to RESPOND):\n{fp_list}")

        assert precision >= 0.95, (
            f"RESPOND precision {precision:.3f} < 0.95. "
            f"False positives: {[m for m, _ in false_positives]}"
        )

    def test_respond_recall_reasonable(self, onnx_svc):
        """RESPOND recall should be reasonable (≥ 0.60).

        We accept lower recall — missing conversational messages is OK (they
        just go through ACT, which is slower but correct). But if recall is
        very low, the model isn't useful as an optimisation.
        """
        hits = 0
        misses = []
        for msg in self.RESPOND_CASES:
            route, conf = _predict(onnx_svc, msg)
            if route == 'respond':
                hits += 1
            else:
                misses.append((msg, conf))

        recall = hits / len(self.RESPOND_CASES) if self.RESPOND_CASES else 0
        logger.info(
            f"RESPOND recall: {recall:.3f} "
            f"({hits}/{len(self.RESPOND_CASES)})"
        )

        if misses:
            miss_list = "\n".join(f"  - {msg!r} (conf={c:.3f})" for msg, c in misses)
            logger.info(f"Missed RESPOND cases (routed to ACT):\n{miss_list}")

        assert recall >= 0.60, (
            f"RESPOND recall {recall:.3f} < 0.60. Model may be too conservative. "
            f"Misses: {[m for m, _ in misses]}"
        )

    def test_act_cases_not_leaked(self, onnx_svc):
        """No more than 2 ACT messages should leak to RESPOND.

        Hard ceiling on false positives for clear-cut action messages.
        """
        leaked = []
        for msg in self.ACT_CASES:
            route, conf = _predict(onnx_svc, msg)
            if route == 'respond':
                leaked.append((msg, conf))

        logger.info(f"ACT→RESPOND leaks: {len(leaked)}/{len(self.ACT_CASES)}")

        if leaked:
            leak_list = "\n".join(f"  - {msg!r} (conf={c:.3f})" for msg, c in leaked)
            logger.warning(f"Leaked ACT messages:\n{leak_list}")

        assert len(leaked) <= 2, (
            f"{len(leaked)} ACT messages leaked to RESPOND (max 2): "
            f"{[m for m, _ in leaked]}"
        )
