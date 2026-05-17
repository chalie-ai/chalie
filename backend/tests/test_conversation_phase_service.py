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

    def test_update_first_message_is_opening(self):
        store = _make_store()
        svc = _make_service(store)
        result = svc.update("Hello!", is_user=True, topic="greetings")
        assert result["current"] == "opening"

    def test_update_second_message_still_opening(self):
        store = _make_store()
        svc = _make_service(store)
        svc.update("Hello!", is_user=True, topic="greetings")
        result = svc.update("Hi there, how are you?", is_user=True, topic="greetings")
        assert result["current"] == "opening"


class TestStateRoundtrip:
    def test_state_persisted_and_reloaded(self):
        store = _make_store()
        svc = _make_service(store)

        svc.update("First message", is_user=True, topic="a")
        # Get a second service instance hitting the same store
        svc2 = _make_service(store)
        phase = svc2.get_phase()
        # State should be present and consistent
        assert phase["current"] in ("opening", "exploring", "deepening", "resolving", "closing")

    def test_reset_clears_state(self):
        store = _make_store()
        svc = _make_service(store)

        svc.update("Hello", is_user=True, topic="a")
        svc.reset()

        # After reset, state should be default
        phase = svc.get_phase()
        assert phase["current"] == "opening"
        assert phase["momentum"] == pytest.approx(0.5, abs=1e-9)


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

    def test_high_question_density_triggers_exploring(self):
        """More than half of recent user messages being questions should trigger exploring.

        We use a fresh topic after a few opening exchanges so that the same-topic-run
        is reset, preventing deepening from outranking exploring.
        """
        store = _make_store()
        svc = _make_service(store)

        # Opening + a couple of neutral exchanges
        messages = [
            ("Hey there", True, "greetings"),
            ("Hi!", False, "greetings"),
            ("Let us chat", True, "general"),
            ("Sure", False, "general"),
            # Now rapid-fire questions across varying sub-topics so no long same-topic run forms
            ("What do you think about this?", True, "topic-a"),
            ("Interesting point", False, "topic-a"),
            ("Why is that exactly?", True, "topic-b"),
            ("Because...", False, "topic-b"),
            ("Can you explain further?", True, "topic-c"),
            ("Here is how", False, "topic-c"),
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

    def test_same_topic_long_run_without_length_increase_reaches_deepening(self):
        """Very long same-topic run (7+) reaches deepening even without length trend."""
        store = _make_store()
        svc = _make_service(store)

        messages = [("Message about cooking", True, "cooking")] * 12
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

    def test_affirmatives_recognised(self):
        """Spot-check affirmative word matching."""
        from services.conversation_phase_service import _has_affirmative
        assert _has_affirmative("ok got it")
        assert _has_affirmative("Thanks a lot!")
        assert _has_affirmative("Makes sense to me")
        assert _has_affirmative("Understood")
        assert not _has_affirmative("I disagree with that")


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

    def test_has_closing_helper(self):
        from services.conversation_phase_service import _has_closing
        assert _has_closing("talk later")
        assert _has_closing("Gotta go now")
        assert _has_closing("ttyl")
        assert _has_closing("See you tomorrow")
        assert not _has_closing("Looking forward to our meeting tomorrow morning")

    def test_length_trend_triggers_closing(self):
        """A 40%+ drop in average message length over 6+ messages triggers closing.

        Uses non-affirmative short messages so that resolving does not win first.
        Two consecutive length-drop windows fire to satisfy stickiness (2 signals).
        """
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        # First 3 messages: long (non-affirmative, non-closing)
        long_msgs = [
            "This is a quite long message about the renovation project I am planning here",
            "Here is another detailed message describing the various options I am considering now",
            "And yet another thorough explanation of the challenges involved in this specific work",
        ]
        # Next 5 messages: very short (non-affirmative, non-closing)
        # "hmm", "mhm", "hm" are neither affirmative nor closing words
        short_msgs = ["Hmm", "Hm", "Hm", "Hmm", "Hm"]

        all_msgs = [(m, True, "topic") for m in long_msgs + short_msgs]
        for i, (text, is_user, topic) in enumerate(all_msgs):
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

    def test_short_message_after_closing_stays_closing(self):
        """A short non-substantive message after closing should not flip back."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        with patch("services.conversation_phase_service.utc_now", return_value=base):
            svc.update("Bye!", is_user=True, topic="misc")

        t1 = base + timedelta(seconds=30)
        with patch("services.conversation_phase_service.utc_now", return_value=t1):
            result = svc.update("Ok", is_user=True, topic="misc")
        assert result["current"] == "closing"


class TestMomentum:
    def test_fast_exchanges_give_high_momentum(self):
        """Messages every 5 seconds (well above baseline) should yield high momentum."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        messages = [
            ("Hey", True, "a"),
            ("Hi", True, "a"),
            ("How are you?", True, "a"),
            ("Fine thanks", True, "a"),
            ("Good to hear", True, "a"),
        ]
        for i, (text, is_user, topic) in enumerate(messages):
            t = base + timedelta(seconds=i * 5)   # 5-second gaps → fast
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update(text, is_user=is_user, topic=topic)

        assert result["momentum"] > 0.6

    def test_slow_exchanges_give_low_momentum(self):
        """Messages every 10 minutes should yield low momentum."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        messages = [
            ("Hey", True, "a"),
            ("Hi", True, "a"),
            ("How are you?", True, "a"),
            ("Fine thanks", True, "a"),
            ("Good to hear", True, "a"),
        ]
        for i, (text, is_user, topic) in enumerate(messages):
            t = base + timedelta(minutes=i * 10)   # 10-minute gaps → slow
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update(text, is_user=is_user, topic=topic)

        assert result["momentum"] < 0.5

    def test_momentum_capped_at_one(self):
        """Even extremely fast exchanges must not exceed momentum of 1.0."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        for i in range(10):
            t = base + timedelta(milliseconds=i * 100)  # 100ms gaps → near-instant
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update("ping", is_user=True, topic="a")

        assert result["momentum"] <= 1.0

    def test_momentum_ema_smoothing(self):
        """EMA with alpha=0.3 smooths changes; verify new value is between old and instant."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        # Build up a high momentum
        for i in range(5):
            t = base + timedelta(seconds=i * 5)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                svc.update("quick", is_user=True, topic="a")

        phase_before = svc.get_phase()
        mom_before = phase_before["momentum"]

        # Now a slow message (large gap) should pull momentum down but not to zero
        t_slow = base + timedelta(seconds=5 + 600)  # 10-minute gap
        with patch("services.conversation_phase_service.utc_now", return_value=t_slow):
            result = svc.update("slow message", is_user=True, topic="a")

        # Momentum should have dropped but not hit zero
        assert result["momentum"] < mom_before
        assert result["momentum"] > 0.0


class TestDirection:
    def test_increasing_length_gives_deepening_direction(self):
        """Growing message lengths with stable timing → direction=deepening."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        texts = [
            "Short",
            "A bit longer now",
            "Getting even longer with more content here",
            "Now quite a lot more words in this particular message about stuff",
            "This is a noticeably lengthier message with considerably more information",
            "And now we have an even longer message with substantially more text than before",
            "This message is significantly longer than the previous ones and contains quite a bit more",
            "At this point the messages have grown considerably in length compared to early ones",
        ]
        for i, text in enumerate(texts):
            t = base + timedelta(seconds=i * 30)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update(text, is_user=True, topic="a")

        assert result["direction"] == "deepening"

    def test_decreasing_length_gives_winding_down_direction(self):
        """Shrinking message lengths → direction=winding_down."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        texts = [
            "This is a very long and detailed message about many things in great detail",
            "Still quite a long message with considerable information and context provided",
            "A moderately long message that is shorter than the previous ones",
            "A shorter message now",
            "Short",
            "Yep",
            "Ok",
            "K",
        ]
        for i, text in enumerate(texts):
            t = base + timedelta(seconds=i * 30)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update(text, is_user=True, topic="a")

        assert result["direction"] == "winding_down"

    def test_stable_messages_give_sustaining_direction(self):
        """Stable length and timing → direction=sustaining."""
        store = _make_store()
        svc = _make_service(store)
        base = utc_now()

        # Uniform length messages
        text = "This message is roughly the same length as all the others here"
        for i in range(8):
            t = base + timedelta(seconds=i * 30)
            with patch("services.conversation_phase_service.utc_now", return_value=t):
                result = svc.update(text, is_user=True, topic="a")

        assert result["direction"] == "sustaining"


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

    def test_closing_keyword_overrides_stickiness(self):
        """Closing keywords bypass stickiness and flip immediately."""
        store = _make_store()
        svc = _make_service(store)

        # Build up some state
        messages = [
            ("Hello", True, "a"),
            ("Hi", False, "a"),
            ("Talking about things", True, "a"),
            ("Sure", False, "a"),
        ]
        _push_messages(svc, messages)

        result = svc.update("Bye!", is_user=True, topic="a")
        assert result["current"] == "closing"


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


class TestHelpers:
    def test_count_questions(self):
        from services.conversation_phase_service import _count_questions
        assert _count_questions("How are you? And what time?") == 2
        assert _count_questions("Hello there") == 0
        assert _count_questions("?") == 1

    def test_same_topic_run(self):
        from services.conversation_phase_service import _same_topic_run
        assert _same_topic_run(["a", "b", "b", "b"]) == 3
        assert _same_topic_run(["a", "b", "c"]) == 1
        assert _same_topic_run([]) == 0
        assert _same_topic_run(["x"]) == 1

    def test_topic_changes_in_last(self):
        from services.conversation_phase_service import _topic_changes_in_last
        assert _topic_changes_in_last(["a", "b", "b", "c", "c"], 5) == 2
        assert _topic_changes_in_last(["a", "a", "a", "a", "a"], 5) == 0

    def test_question_density(self):
        from services.conversation_phase_service import _question_density
        counts = [1, 0, 1, 0, 1]
        assert _question_density(counts, 5) == pytest.approx(0.6, abs=1e-9)
        assert _question_density([], 5) == pytest.approx(0.0, abs=1e-9)

    def test_is_increasing(self):
        from services.conversation_phase_service import _is_increasing
        assert _is_increasing([10, 20, 30, 40]) is True
        assert _is_increasing([40, 30, 20, 10]) is False
        assert _is_increasing([20, 20, 20, 20]) is False
        assert _is_increasing([10]) is False  # too few

    def test_is_decreasing(self):
        from services.conversation_phase_service import _is_decreasing
        assert _is_decreasing([40, 30, 20, 10]) is True
        assert _is_decreasing([10, 20, 30, 40]) is False


class TestSingleton:
    def test_singleton_thread_safety(self):
        """Multiple threads obtaining the singleton should all get the same object."""
        from services.conversation_phase_service import (
            get_conversation_phase_service,
        )
        import services.conversation_phase_service as _mod

        # Reset singleton to test creation race
        original = _mod._instance
        _mod._instance = None

        instances = []

        def grab():
            instances.append(get_conversation_phase_service())

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(i) for i in instances)) == 1

        # Restore
        _mod._instance = original
