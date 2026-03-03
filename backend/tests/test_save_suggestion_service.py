"""Tests for SaveSuggestionService — detection heuristics, flag lifecycle,
trigger signals, rate limiting, and document creation flow."""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from services.save_suggestion_service import SaveSuggestionService

pytestmark = pytest.mark.unit


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def service():
    return SaveSuggestionService()


@pytest.fixture
def mock_store():
    """Provide a mock MemoryStore that default-returns no existing keys."""
    store = MagicMock()
    store.exists.return_value = False
    store.get.return_value = None
    return store


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
            result = service.detect_saveable_content(WORKOUT_PLAN, 'fitness', 'thread1')

        assert result is not None
        assert result['content_type'] == 'plan'

    def test_detects_day_by_day_itinerary(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(DAY_BY_DAY_PLAN, 'travel', 'thread1')

        assert result is not None
        assert result['content_type'] == 'plan'

    def test_rejects_short_response(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(SHORT_RESPONSE, 'chat', 'thread1')

        assert result is None


# ── Detection: Recipes ───────────────────────────────────────

class TestDetectRecipe:

    def test_detects_recipe(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(RECIPE, 'cooking', 'thread1')

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
            result = service.detect_saveable_content(no_qty, 'cooking', 'thread1')

        # Should not match: too short (< 300 chars)
        assert result is None


# ── Detection: Structured Lists ──────────────────────────────

class TestDetectStructuredList:

    def test_detects_gear_list(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(STRUCTURED_LIST, 'camping', 'thread1')

        assert result is not None
        assert result['content_type'] == 'list'

    def test_rejects_inline_list(self, service, mock_store):
        """A short inline list without headers should not match."""
        short_list = "Here are some options:\n- Option A\n- Option B\n- Option C"
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(short_list, 'chat', 'thread1')

        assert result is None


# ── Detection: False Positive Guards ─────────────────────────

class TestFalsePositiveGuards:

    def test_conversational_short_response_rejected(self, service, mock_store):
        """Short conversational openers don't trigger save."""
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(CONVERSATIONAL, 'chat', 'thread1')

        assert result is None

    def test_cooldown_prevents_detection(self, service, mock_store):
        """If cooldown key exists, detection returns None."""
        mock_store.exists.return_value = True  # cooldown exists
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.detect_saveable_content(WORKOUT_PLAN, 'fitness', 'thread1')

        assert result is None


# ── Flag Lifecycle ───────────────────────────────────────────

class TestFlagLifecycle:

    def test_flag_set_and_get(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            service.flag_saveable('thread1', 'fitness', 'plan', 'ex123')

        mock_store.setex.assert_called_once()
        call_args = mock_store.setex.call_args
        assert call_args[0][0] == 'saveable:thread1'
        assert call_args[0][1] == 1800  # 30min TTL
        data = json.loads(call_args[0][2])
        assert data['content_type'] == 'plan'
        assert data['exchange_id'] == 'ex123'

    def test_get_flag_returns_data(self, service, mock_store):
        flag_data = json.dumps({'content_type': 'plan', 'topic': 'fitness', 'ts': 123})
        mock_store.get.return_value = flag_data
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.get_saveable_flag('thread1')

        assert result is not None
        assert result['content_type'] == 'plan'

    def test_get_flag_returns_none_when_missing(self, service, mock_store):
        mock_store.get.return_value = None
        with patch.object(service, '_get_store', return_value=mock_store):
            result = service.get_saveable_flag('thread1')

        assert result is None

    def test_clear_flag(self, service, mock_store):
        with patch.object(service, '_get_store', return_value=mock_store):
            service.clear_flag('thread1')

        mock_store.delete.assert_called_once_with('saveable:thread1')


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
        with patch.object(service, '_get_store', return_value=mock_store):
            service.record_rejection('thread1', 'fitness')

        # Should set both cooldown and topic rejection keys
        assert mock_store.setex.call_count == 2
        keys_set = [call[0][0] for call in mock_store.setex.call_args_list]
        assert 'save_suggest:cooldown:thread1' in keys_set
        assert 'save_suggest:reject:thread1:fitness' in keys_set

    def test_duplicate_prevention(self, service, mock_store):
        """First call returns False (not duplicate), second returns True."""
        mock_store.exists.side_effect = [False, True]
        with patch.object(service, '_get_store', return_value=mock_store):
            assert service._is_duplicate('hash123') is False
            assert service._is_duplicate('hash123') is True


# ── Document Creation Flow ───────────────────────────────────

class TestDocumentCreation:

    def test_create_document_full_flow(self, service, mock_store):
        """Test the full create flow: conversation → synthesis → document."""
        mock_store.exists.return_value = False  # no duplicate

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
                 patch('services.document_service.DocumentService', return_value=mock_doc_svc):
                doc_id = service.create_document_from_conversation('thread1', 'fitness', 'plan')

        assert doc_id == 'abc12345'
        mock_doc_svc.create_document_from_text.assert_called_once()

    def test_create_document_empty_conversation(self, service, mock_store):
        """Returns None if no conversation content found."""
        with patch.object(service, '_get_store', return_value=mock_store), \
             patch.object(service, '_get_conversation_window', return_value=None):
            result = service.create_document_from_conversation('thread1', 'topic', 'plan')

        assert result is None

    def test_create_document_duplicate_skipped(self, service, mock_store):
        """Returns None if duplicate conversation hash detected."""
        mock_store.exists.return_value = True  # duplicate exists

        with patch.object(service, '_get_store', return_value=mock_store), \
             patch.object(service, '_get_conversation_window', return_value="User: test\n\nAssistant: test response"):
            result = service.create_document_from_conversation('thread1', 'topic', 'plan')

        assert result is None


# ── Card Emission ────────────────────────────────────────────

class TestCardEmission:

    def test_emit_save_card_calls_document_card_service(self, service):
        mock_card = MagicMock()
        with patch('services.document_card_service.DocumentCardService', return_value=mock_card):
            service.emit_save_card('thread1', 'fitness', 'plan')

            mock_card.emit_save_suggestion_card.assert_called_once_with(
                'fitness', 'plan', 'thread1',
            )
