"""Tests for SaveSuggestionService — detection heuristics, flag lifecycle,
trigger signals, rate limiting, and document creation flow."""

import hashlib
import json
import pytest
from unittest.mock import patch, MagicMock

from services.save_suggestion_service import SaveSuggestionService
from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def service():
    return SaveSuggestionService()


@pytest.fixture
def mock_store():
    """Provide a real MemoryStore for save-suggestion tests.

    A fresh empty store already returns ``None`` for ``get()`` and ``False``
    for ``exists()``, matching the previous MagicMock defaults.

    # TODO: rename fixture to ``store`` after all references migrated.
    """
    return MemoryStore()


# ── Sample content ────────────────────────────────────────────

WORKOUT_PLAN = """## 4-Week Workout Plan

### Week 1: Foundation
1. Monday: Upper body — 3x12 push-ups, 3x10 rows, 3x15 curls
2. Tuesday: Lower body — 3x15 squats, 3x12 lunges, 3x20 calf raises
3. Wednesday: Rest
4. Thursday: Full body — circuit training
5. Friday: Cardio — 30 min jog
6. Saturday: Flexibility — yoga session
7. Sunday: Rest

### Week 2: Progression
1. Monday: Upper body — increase weight by 5%
2. Tuesday: Lower body — add jump squats
3. Wednesday: Active recovery
4. Thursday: Full body — HIIT
5. Friday: Cardio — interval sprints
"""

RECIPE = """## Grandma's Chocolate Chip Cookies

### Ingredients
- 2 cups all-purpose flour
- 1 tsp baking soda
- 1 tsp salt
- 1 cup butter, softened
- 3/4 cup sugar
- 3/4 cup brown sugar
- 2 large eggs
- 2 tsp vanilla extract
- 2 cups chocolate chips

### Instructions
1. Preheat oven to 375°F
2. Mix flour, baking soda, and salt in a bowl
3. Cream butter and sugars until fluffy
4. Beat in eggs and vanilla
5. Gradually stir in flour mixture
6. Fold in chocolate chips
7. Drop by spoonfuls onto baking sheet
8. Bake 9-11 minutes until golden
"""

STRUCTURED_LIST = """## Essential Camping Gear List

### Shelter & Sleep
- Tent (3-season, 2-person)
- Sleeping bag (rated to 30°F)
- Sleeping pad
- Ground tarp

### Cooking & Food
- Camp stove
- Fuel canisters
- Cookware set
- Water filter
- Cooler

### Safety & Navigation
- First aid kit
- Headlamp
- Map and compass
- Emergency whistle
- Fire starter
"""

SHORT_RESPONSE = "Sure, I can help with that!"

CONVERSATIONAL = "Here's a quick summary of what we discussed earlier."

DAY_BY_DAY_PLAN = """## 7-Day Travel Itinerary — Tokyo

### Day 1: Arrival & Shinjuku
Arrive at Narita Airport. Take the N'EX train to Shinjuku.

### Day 2: Traditional Tokyo
Visit Senso-ji temple in Asakusa, then explore Ueno Park.

### Day 3: Pop Culture
Akihabara for electronics and anime, Harajuku for street fashion.

### Day 4: Day Trip to Kamakura
Great Buddha, Hase-dera temple, beach walk.
"""


# ── Detection: Plans ─────────────────────────────────────────

class TestDetectPlan:

    def test_detects_workout_plan(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(WORKOUT_PLAN, 'fitness')

        assert result is not None
        assert result['content_type'] == 'plan'

    def test_detects_day_by_day_itinerary(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(DAY_BY_DAY_PLAN, 'travel')

        assert result is not None
        assert result['content_type'] == 'plan'

    def test_rejects_short_response(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(SHORT_RESPONSE, 'chat')

        assert result is None


# ── Detection: Recipes ───────────────────────────────────────

class TestDetectRecipe:

    def test_detects_recipe(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(RECIPE, 'cooking')

        assert result is not None
        assert result['content_type'] == 'recipe'

    def test_recipe_needs_quantities(self, service, mock_store):
        """Recipe detection requires quantity patterns, not just headers."""
        no_qty = """## My Recipe

### Ingredients
- flour
- sugar
- eggs

### Instructions
1. Mix everything
2. Bake
"""
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(no_qty, 'cooking')

        # Should not match: too short (< 300 chars)
        assert result is None


# ── Detection: Structured Lists ──────────────────────────────

class TestDetectStructuredList:

    def test_detects_gear_list(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(STRUCTURED_LIST, 'camping')

        assert result is not None
        assert result['content_type'] == 'list'

    def test_rejects_inline_list(self, service, mock_store):
        """A short inline list without headers should not match."""
        short_list = "Here are some options:\n- Option A\n- Option B\n- Option C"
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(short_list, 'chat')

        assert result is None


# ── Detection: False Positive Guards ─────────────────────────

class TestFalsePositiveGuards:

    def test_conversational_short_response_rejected(self, service, mock_store):
        """Short conversational openers don't trigger save."""
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(CONVERSATIONAL, 'chat')

        assert result is None

    def test_cooldown_prevents_detection(self, service, mock_store):
        """If cooldown key exists, detection returns None."""
        mock_store.set('save_suggest:cooldown', '1')  # pre-populate cooldown key
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(WORKOUT_PLAN, 'fitness')

        assert result is None


# ── Flag Lifecycle ───────────────────────────────────────────

class TestFlagLifecycle:

    def test_flag_set_and_get(self, service, mock_store):
        """``flag_saveable`` writes flag JSON with correct key, TTL, and payload."""
        with patch.object(service, '_get_store', return_value=mock_store):
            service.flag_saveable('fitness', 'plan', 'ex123')

        raw = mock_store.get('saveable')
        assert raw is not None, "flag key must exist in store after flag_saveable"
        data = json.loads(raw)
        assert data['content_type'] == 'plan'
        assert data['exchange_id'] == 'ex123'
        assert 0 < mock_store.ttl('saveable') <= 1800  # 30-min TTL

    def test_get_flag_returns_data(self, service, mock_store):
        """``get_saveable_flag`` deserialises an existing flag from the store."""
        flag_data = json.dumps({'content_type': 'plan', 'topic': 'fitness', 'ts': 123})
        mock_store.set('saveable', flag_data)
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.get_saveable_flag()

        assert result is not None
        assert result['content_type'] == 'plan'

    def test_get_flag_returns_none_when_missing(self, service, mock_store):
        """``get_saveable_flag`` returns None when no flag key exists."""
        # fresh empty store → get() returns None by default
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.get_saveable_flag()

        assert result is None

    def test_clear_flag(self, service, mock_store):
        """``clear_flag`` removes the saveable flag key from the store."""
        mock_store.set('saveable', 'some_data')
        with patch.object(service, '_get_store', return_value=mock_store):
            service.clear_flag()

        assert mock_store.get('saveable') is None


# ── Trigger Signal Detection ─────────────────────────────────

class TestTriggerDetection:

    @pytest.mark.parametrize("text", [
        "Looks good!",
        "Perfect, thanks!",
        "That's great",
        "Love it",
        "This is exactly what I needed",
        "Thanks, done",
        "thanks!",
    ])
    def test_completion_signals(self, service, text):
        assert service.detect_save_trigger(text) == 'completion'

    @pytest.mark.parametrize("text", [
        "Save this for later",
        "I'll work on this later",
        "Keep this",
        "save it",
    ])
    def test_deferral_signals(self, service, text):
        assert service.detect_save_trigger(text) == 'deferral'

    @pytest.mark.parametrize("text", [
        "Can you modify step 3?",
        "What about adding more exercises?",
        "I don't think that's right",
        "Change the recipe to be vegan",
        "",
        None,
    ])
    def test_non_trigger_messages(self, service, text):
        assert service.detect_save_trigger(text) is None


# ── Rate Limiting ────────────────────────────────────────────

class TestRateLimiting:

    def test_record_rejection_sets_cooldown_and_reject(self, service, mock_store):
        """``record_rejection`` persists both the per-thread cooldown and the topic-rejection keys."""
        with patch.object(service, '_get_store', return_value=mock_store):
            service.record_rejection('fitness')

        # Both keys must exist with non-zero TTLs after a rejection
        assert mock_store.exists('save_suggest:cooldown'), "cooldown key must be set"
        assert mock_store.exists('save_suggest:reject:fitness'), "topic reject key must be set"

    def test_duplicate_prevention(self, service, mock_store):
        """First call returns False (not duplicate); second call returns True (already seen)."""
        # The real MemoryStore sets the hash key on the first call so the second
        # call naturally finds it — no side_effect manipulation required.
        with patch.object(service, '_get_store', return_value=mock_store):
            assert service._is_duplicate('hash123') is False
            assert service._is_duplicate('hash123') is True


# ── Document Creation Flow ───────────────────────────────────

class TestDocumentCreation:

    def test_create_document_full_flow(self, service, mock_store):
        """Test the full create flow: conversation → synthesis → document."""
        # fresh empty store → no duplicate (exists returns False by default)

        turns = [
            {'role': 'user', 'content': 'Create a workout plan'},
            {'role': 'assistant', 'content': WORKOUT_PLAN},
        ]
        mock_wm = MagicMock()
        mock_wm.get_recent_turns.return_value = turns

        mock_llm_response = MagicMock()
        mock_llm_response.text = "# Workout Plan\n\nGenerated content..."

        mock_doc_svc = MagicMock()
        mock_doc_svc.create_document_from_text.return_value = 'abc12345'

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = mock_llm_response

        with patch.object(service, '_get_store', return_value=mock_store), \
             patch.object(service, '_get_conversation_window', return_value="User: Create a workout plan\n\nAssistant: " + WORKOUT_PLAN), \
             patch.object(service, '_synthesize_document', return_value="# Workout Plan\n\nGenerated content..."):

            with patch('services.database_service.get_shared_db_service'), \
                 patch('services.document_service.DocumentService', return_value=mock_doc_svc), \
                 patch('services.innate_skills.document_skill.create_document_artifacts', return_value=3):
                doc_id = service.create_document_from_conversation('fitness', 'plan')

        assert doc_id == 'abc12345'
        mock_doc_svc.create_document_from_text.assert_called_once()

    def test_create_document_empty_conversation(self, service, mock_store):
        """Returns None if no conversation content found."""
        with patch.object(service, '_get_store', return_value=mock_store), \
             patch.object(service, '_get_conversation_window', return_value=None):
            result = service.create_document_from_conversation('topic', 'plan')

        assert result is None

    def test_create_document_duplicate_skipped(self, service, mock_store):
        """Returns None if duplicate conversation hash detected."""
        conv = "User: test\n\nAssistant: test response"
        dup_key = f"save_suggest:hash:{hashlib.sha256(conv.encode()).hexdigest()}"
        mock_store.set(dup_key, '1')  # pre-populate duplicate hash so first call sees it

        with patch.object(service, '_get_store', return_value=mock_store), \
             patch.object(service, '_get_conversation_window', return_value=conv):
            result = service.create_document_from_conversation('topic', 'plan')

        assert result is None
