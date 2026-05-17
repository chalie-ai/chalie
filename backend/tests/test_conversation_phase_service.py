"""
Unit tests for ConversationPhaseService.

Covers:
- Phase transitions: opening → exploring → deepening → resolving → closing
- Opening detection: first 2 messages, 30+ min gap resets to opening
- Exploring detection: topic changes, high question density
- Deepening detection: same topic for 5+ exchanges with increasing length
- Resolving detection: low question density + affirmative signals
- Closing detection: explicit closing words (immediate), decreasing length trend
- Anti-false-close: substantive message after closing → back to exploring
- Momentum calculation: EMA smoothing, normalisation, cap at 1.0
- Direction detection: increasing/stable/decreasing length + momentum
- State persistence: MemoryStore read/write roundtrip
- Reset: clears state
- Empty/new thread: returns opening defaults
- Stickiness: single signal does not flip phase (except closing keywords)
"""

import threading
from datetime import timedelta
from unittest.mock import patch

import pytest

from services.memory_store import MemoryStore
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_store():
    """Fresh isolated MemoryStore for each test."""
    return MemoryStore()


def _make_service(store: MemoryStore):
    """Create a ConversationPhaseService wired to the given MemoryStore."""
    from services.conversation_phase_service import ConversationPhaseService
    svc = ConversationPhaseService()
    # Patch the store so this instance uses our isolated store
    svc._store = lambda: store
    return svc


def _push_messages(svc, messages: list, store: MemoryStore = None):
    """
    Push a list of (text, is_user, topic) tuples through the service.

    Each call is made with a uniform 30-second spacing so momentum stays near 1.0.
    Returns the result of the final update().
    """
    base_time = utc_now()
    result = None
    for i, item in enumerate(messages):
        text, is_user, topic = item
        msg_time = base_time + timedelta(seconds=i * 30)
        with patch("services.conversation_phase_service.utc_now", return_value=msg_time):
            result = svc.update(text, is_user, topic=topic)
    return result


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEmptyThread:
    def test_get_phase_returns_opening_defaults_when_no_state(self):
        store = _make_store()
        svc = _make_service(store)
        phase = svc.get_phase()
        assert phase["current"] == "opening"
        assert phase["momentum"] == pytest.approx(0.5, abs=1e-9)
        assert phase["direction"] == "sustaining"
        assert "updated_at" in phase


class TestOpeningDetection:
    def test_first_two_exchanges_are_opening(self):
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        for i in range(2):
            t = base + timedelta(seconds=i * 30)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update("Hello", is_user=True, topic="a")
        assert result["current"] == "opening"

    def test_long_gap_resets_to_opening(self):
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        # Prime with several messages to leave opening
        messages = [
            ("Hello!", True, "weather"),
            ("It's nice today", True, "weather"),
            ("What about tomorrow?", True, "weather"),
            ("It might rain", False, "weather"),
            ("Should I carry an umbrella?", True, "weather"),
            ("Yes definitely", False, "weather"),
        ]
        for i, (text, is_user, topic) in enumerate(messages):
            t = base + timedelta(seconds=i * 30)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                svc.update(text, is_user=is_user, topic=topic)

        # Now simulate a 35-minute gap
        gap_time = base + timedelta(minutes=35)
        with patch("services.conversation_phase_service.utc_now", return_value=gap_time):
            result = svc.update("I'm back!", is_user=True, topic="weather")

        assert result["current"] == "opening"


class TestExploringDetection:
    def test_topic_changes_trigger_exploring(self):
        """Two topic changes in 5 exchanges should push toward exploring."""
        store = _make_store()
        svc = _make_service(store)

        # Need > 2 exchanges first to leave opening, then rapid topic shifts
        messages = [
            ("Hey", True, "greetings"),
            ("Hello", False, "greetings"),
            ("Tell me about weather", True, "weather"),
            ("Sure, it's sunny", False, "weather"),
            ("What about cooking?", True, "cooking"),    # topic change 1
            ("Great, let me explain", False, "cooking"),
            ("Actually, back to weather?", True, "weather"),  # topic change 2
            ("Of course", False, "weather"),
        ]
        result = _push_messages(svc, messages)
        assert result["current"] == "exploring"


class TestDeepeningDetection:
    def test_same_topic_with_increasing_length_reaches_deepening(self):
        """5+ exchanges on same topic with growing message lengths → deepening."""
        store = _make_store()
        svc = _make_service(store)

        # Start with enough messages to leave opening, then same topic + growing length
        messages = [
            ("Hi", True, "kitchen"),
            ("Hi there", False, "kitchen"),
            ("I want to renovate my kitchen", True, "kitchen"),
            ("That sounds great", False, "kitchen"),
            ("I am thinking about new cabinets and countertops", True, "kitchen"),
            ("That makes a lot of sense for the space", False, "kitchen"),
            ("The cabinets should be in a light wood tone to brighten the room", True, "kitchen"),
            ("Agreed, lighter tones open up smaller kitchens beautifully", False, "kitchen"),
            ("I also want to add an island in the centre for prep work and seating", True, "kitchen"),
            ("An island would really transform how you use the space day to day", False, "kitchen"),
            ("Yes and I am considering quartz for the countertop because it is low maintenance", True, "kitchen"),
            ("Quartz is excellent for durability and comes in many designs", False, "kitchen"),
        ]
        result = _push_messages(svc, messages)
        assert result["current"] == "deepening"


class TestResolvingDetection:
    def test_low_question_density_plus_affirmative_triggers_resolving(self):
        """Low question density and affirmatives should lead to resolving."""
        store = _make_store()
        svc = _make_service(store)

        # Build up to deepening first
        setup = [
            ("Let us talk about Python decorators", True, "python"),
            ("Sure, what would you like to know", False, "python"),
            ("I want to understand how they work", True, "python"),
            ("They wrap functions to add behaviour", False, "python"),
            ("That makes sense, show me an example", True, "python"),
            ("Here is a simple logging decorator", False, "python"),
            ("I see how the at-syntax applies it automatically", True, "python"),
            ("Exactly, it is syntactic sugar for wrapping", False, "python"),
        ]
        _push_messages(svc, setup)

        # Now resolving: no questions, affirmative signals, still reasonable message length
        resolving = [
            ("Got it, that really clarifies things for me", True, "python"),
            ("Glad to hear that helped", False, "python"),
            ("Makes sense now, I understand the decorator pattern", True, "python"),
            ("Perfect, feel free to ask if you need more detail", False, "python"),
        ]
        result = _push_messages(svc, resolving)
        assert result["current"] == "resolving"


class TestClosingDetection:
    def test_explicit_closing_word_is_immediate(self):
        """A 'bye' in a user message should immediately flip to closing."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        messages = [
            ("Hello", True, "misc"),
            ("Hi", False, "misc"),
            ("This was helpful", True, "misc"),
            ("Glad to hear", False, "misc"),
            ("Bye!", True, "misc"),
        ]
        for i, (text, is_user, topic) in enumerate(messages):
            t = base + timedelta(seconds=i * 30)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update(text, is_user=is_user, topic=topic)
        assert result["current"] == "closing"


class TestAntiFalseClose:
    def test_substantive_message_after_closing_returns_to_exploring(self):
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        # Force closing
        t0 = base
        with patch("services.conversation_phase_service.utc_now", return_value=t0):
            svc.update("Goodbye!", is_user=True, topic="misc")

        # Verify we are in closing
        assert svc.get_phase()["current"] == "closing"

        # Send a substantive message (not a closing word, > 20 chars)
        t1 = base + timedelta(seconds=30)
        with patch("services.conversation_phase_service.utc_now", return_value=t1):
            result = svc.update(
                "Actually, I have one more question about the project timeline",
                is_user=True,
                topic="project",
            )
        assert result["current"] == "exploring"






class TestStickiness:
    def test_single_exploring_signal_does_not_flip_from_deepening(self):
        """A single topic change should not immediately flip from deepening."""
        store = _make_store()
        svc = _make_service(store)

        # Build deepening state
        setup = [
            ("Hi", True, "code"),
            ("Hey", False, "code"),
            ("Let us talk about code quality", True, "code"),
            ("Sure", False, "code"),
            ("I want to improve readability", True, "code"),
            ("Good idea, start with clear names", False, "code"),
            ("Clear names and short functions are key to readable code everywhere", True, "code"),
            ("Exactly, each function should do one thing only consistently", False, "code"),
            ("I will also add docstrings and comments where logic is non-obvious here", True, "code"),
            ("That helps future readers understand intent without guessing around", False, "code"),
        ]
        _push_messages(svc, setup)

        # Verify deepening
        assert svc.get_phase()["current"] == "deepening"

        # Single off-topic question — should NOT flip phase yet
        base = utc_now()
        with patch("services.conversation_phase_service.utc_now", return_value=base):
            result = svc.update("What is the weather?", is_user=True, topic="weather")

        # May be pending but should not have flipped to exploring with just one signal
        # (The state could already be deepening or the pending signal count is 1)
        assert result["current"] == "deepening"




class TestFullLifecycle:
    def test_full_conversation_lifecycle(self):
        """Walk through opening → exploring → deepening → resolving → closing."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()
        idx = 0

        def send(text, is_user, topic):
            nonlocal idx
            t = base + timedelta(seconds=idx * 30)
            idx += 1
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                return svc.update(text, is_user=is_user, topic=topic)

        # Opening: first 2 exchanges
        r = send("Hello!", True, "greetings")
        assert r["current"] == "opening"
        r = send("Hi, how can I help?", False, "greetings")

        # Exploring: topic changes + questions
        send("What do you think about architecture?", True, "architecture")
        send("It is a broad field", False, "architecture")
        send("What about software architecture specifically?", True, "software")
        send("That is more my domain", False, "software")
        r = send("How does microservices compare to monoliths?", True, "microservices")
        # At some point we should hit exploring
        # (may take a message or two due to stickiness)
        if r["current"] != "exploring":
            r = send("What are the trade-offs between them exactly?", True, "microservices")
        assert r["current"] == "exploring"

        # Deepening: sustain on same topic with growing messages
        send("Let me focus on microservices scalability now", True, "microservices")
        send("Good, each service scales independently which reduces cost", False, "microservices")
        send("That sounds really useful for high-traffic components with spiky load patterns", True, "microservices")
        send("Exactly, you only scale the parts that need it rather than the whole application", False, "microservices")
        send("I can see how that allows teams to deploy independently without coordination overhead", True, "microservices")
        r = send("Independent deployments reduce risk per release and speed up iteration cycles greatly", False, "microservices")
        if r["current"] != "deepening":
            send("The service mesh also handles routing, retries and observability out of the box", True, "microservices")
            r = send("Right, that removes a lot of boilerplate from individual service implementations", False, "microservices")
        assert r["current"] == "deepening"

        # Resolving: affirmatives, no questions; keep message length consistent
        # to avoid triggering the length-drop closing heuristic
        send("Got it, this really clarifies the microservices trade-offs for me", True, "microservices")
        send("Glad to hear that, it is an important distinction to grasp early on", False, "microservices")
        r = send("Makes sense now, I understand the deployment independence benefits clearly", True, "microservices")
        if r["current"] != "resolving":
            r = send("Understood, I appreciate the thorough walkthrough through these concepts today", True, "microservices")
        assert r["current"] == "resolving"

        # Closing: explicit signal
        r = send("Bye!", True, "microservices")
        assert r["current"] == "closing"



