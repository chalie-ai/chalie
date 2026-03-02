"""
Save Suggestion Service — Detects saveable content in conversations and suggests
persistence via interactive card.

Architecture:
  1. Phase D (post-response): Deterministic heuristic detects structured deliverables
     in the assistant's response. If found, sets a Redis flag.
  2. Phase A (next message): If user sends a completion/deferral signal and a flag
     exists, emit a save suggestion card.
  3. Idle trigger: Thread expiry service checks for stale flags (5min idle) and
     emits a save suggestion card.

V1 scope: Plans/guides, recipes, structured lists. Expand via telemetry.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SAVE SUGGEST]"

# Redis key patterns
FLAG_KEY = "saveable:{thread_id}"
COOLDOWN_KEY = "save_suggest:cooldown:{thread_id}"
REJECT_KEY = "save_suggest:reject:{thread_id}:{topic}"

# TTLs
FLAG_TTL = 1800           # 30 minutes
COOLDOWN_TTL = 3600       # 1 hour
REJECT_TTL = 604800       # 7 days

# Detection thresholds
MIN_RESPONSE_LENGTH = 300


class SaveSuggestionService:
    """Detects saveable conversation content and orchestrates save suggestions."""

    # ──────────────────────────────────────────────────────────────────────────
    # Detection (called from digest_worker Phase D)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_saveable_content(
        self,
        response_text: str,
        topic: str,
        thread_id: str,
    ) -> Optional[Dict[str, str]]:
        """
        Deterministic heuristic: check if the response contains a finalized artifact.

        Returns {'content_type': str, 'confidence': str} or None.

        V1 artifact types:
          1. Plans/guides — numbered step sequences, day-by-day structure
          2. Recipes — Ingredients + Instructions structure
          3. Structured lists — titled checklists with 5+ items
        """
        if not response_text or len(response_text) < MIN_RESPONSE_LENGTH:
            return None

        # Guard: skip conversational openers
        conversational_starts = (
            'sure,', 'here\'s', 'here is', 'of course', 'absolutely',
            'great question', 'good question', 'i\'d be happy', 'let me',
        )
        first_line = response_text.strip().split('\n')[0].lower().strip()
        # Only skip if the ENTIRE response is short (i.e., it's just a conversational reply)
        # Long responses that start conversationally still contain deliverables
        if len(response_text) < 500 and first_line.startswith(conversational_starts):
            return None

        # Guard: cooldown check
        try:
            redis = self._get_redis()
            cooldown_key = COOLDOWN_KEY.format(thread_id=thread_id)
            if redis.exists(cooldown_key):
                return None
        except Exception:
            pass

        # Guard: topic rejection check
        try:
            redis = self._get_redis()
            reject_key = REJECT_KEY.format(thread_id=thread_id, topic=topic or 'unknown')
            if redis.exists(reject_key):
                return None
        except Exception:
            pass

        # Check artifact types (order matters: more specific first)
        result = self._detect_recipe(response_text)
        if result:
            return result

        result = self._detect_plan(response_text)
        if result:
            return result

        result = self._detect_structured_list(response_text)
        if result:
            return result

        return None

    def _detect_plan(self, text: str) -> Optional[Dict[str, str]]:
        """Detect plans/guides: numbered steps, day-by-day, weekly structure."""
        lower = text.lower()

        # Pattern 1: Header + numbered steps (5+)
        numbered_steps = re.findall(r'^\s*\d+[\.\)]\s+', text, re.MULTILINE)
        has_plan_header = bool(re.search(
            r'(?:^|\n)#{1,3}\s+.*(?:plan|guide|routine|schedule|itinerary|program|workout|checklist)',
            lower,
        ))

        if len(numbered_steps) >= 5 and has_plan_header:
            return {'content_type': 'plan', 'confidence': 'high'}

        # Pattern 2: Day-by-day or week-by-week structure
        day_headers = re.findall(
            r'(?:^|\n)#{1,4}\s+(?:day|week|month|phase|stage)\s+\d',
            lower,
        )
        if len(day_headers) >= 3:
            return {'content_type': 'plan', 'confidence': 'high'}

        # Pattern 3: Many numbered steps without explicit plan header
        if len(numbered_steps) >= 8:
            return {'content_type': 'plan', 'confidence': 'medium'}

        return None

    def _detect_recipe(self, text: str) -> Optional[Dict[str, str]]:
        """Detect recipes: Ingredients + Instructions sections."""
        lower = text.lower()

        has_ingredients = bool(re.search(
            r'(?:^|\n)#{1,4}\s*ingredients', lower,
        ))
        has_instructions = bool(re.search(
            r'(?:^|\n)#{1,4}\s*(?:instructions|steps|directions|method|preparation)',
            lower,
        ))

        if has_ingredients and has_instructions:
            # Confirm with quantity patterns
            quantities = re.findall(
                r'\d+\s*(?:cups?|tbsp|tsp|oz|g|kg|ml|lb|tablespoons?|teaspoons?|cloves?|pieces?)',
                lower,
            )
            if len(quantities) >= 2:
                return {'content_type': 'recipe', 'confidence': 'high'}

        return None

    def _detect_structured_list(self, text: str) -> Optional[Dict[str, str]]:
        """Detect structured lists: titled sections with 5+ items."""
        lower = text.lower()

        # Must have at least one header
        headers = re.findall(r'(?:^|\n)#{1,4}\s+.+', text)
        if not headers:
            return None

        # Count list items (- or * bullets, or numbered items)
        list_items = re.findall(r'^\s*[-*]\s+.+', text, re.MULTILINE)
        numbered_items = re.findall(r'^\s*\d+[\.\)]\s+.+', text, re.MULTILINE)
        total_items = len(list_items) + len(numbered_items)

        # Exclude if it looks like a conversation turn list
        if total_items < 5:
            return None

        # Check for list-type headers
        has_list_header = bool(re.search(
            r'(?:^|\n)#{1,4}\s+.*(?:list|items|things|tools|resources|supplies|requirements|recommendations)',
            lower,
        ))

        if has_list_header and total_items >= 5:
            return {'content_type': 'list', 'confidence': 'high'}

        if total_items >= 10 and len(headers) >= 2:
            return {'content_type': 'list', 'confidence': 'medium'}

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Flagging
    # ──────────────────────────────────────────────────────────────────────────

    def flag_saveable(
        self,
        thread_id: str,
        topic: str,
        content_type: str,
        exchange_id: str,
    ) -> None:
        """Set Redis flag: saveable:{thread_id}. TTL 30min."""
        try:
            redis = self._get_redis()
            key = FLAG_KEY.format(thread_id=thread_id)
            data = json.dumps({
                'content_type': content_type,
                'topic': topic or 'unknown',
                'exchange_id': exchange_id,
                'ts': time.time(),
            })
            redis.setex(key, FLAG_TTL, data)
            logger.info(f"{LOG_PREFIX} Flagged saveable {content_type} in thread {thread_id}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} flag_saveable failed: {e}")

    def get_saveable_flag(self, thread_id: str) -> Optional[Dict]:
        """Read and return the saveable flag if it exists."""
        try:
            redis = self._get_redis()
            key = FLAG_KEY.format(thread_id=thread_id)
            raw = redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_saveable_flag failed: {e}")
        return None

    def clear_flag(self, thread_id: str) -> None:
        """Remove the saveable flag."""
        try:
            redis = self._get_redis()
            key = FLAG_KEY.format(thread_id=thread_id)
            redis.delete(key)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} clear_flag failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Trigger detection (called from digest_worker Phase A on next message)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_save_trigger(self, user_text: str) -> Optional[str]:
        """
        Check if user message signals completion or deferral.
        Returns trigger type: 'completion', 'deferral', or None.
        """
        if not user_text:
            return None

        lower = user_text.lower().strip()

        # Completion signals
        completion_patterns = [
            r'\b(?:looks?\s+good|perfect|that\'?s?\s+great|love\s+it|excellent|awesome)\b',
            r'\b(?:exactly\s+what\s+i\s+(?:needed|wanted)|this\s+is\s+(?:great|perfect|exactly))\b',
            r'\b(?:thanks?,?\s*(?:done|that\'?s?\s+it|all\s+good|i\'?m?\s+set))\b',
            r'\b(?:great,?\s+thanks?|nice\s+work|well\s+done)\b',
            r'^(?:thanks?!?|ty!?|thx!?)$',
        ]
        for pattern in completion_patterns:
            if re.search(pattern, lower):
                return 'completion'

        # Deferral signals
        deferral_patterns = [
            r'\b(?:save\s+(?:this|it)|keep\s+(?:this|it)|store\s+(?:this|it))\b',
            r'\b(?:(?:i\'?ll|will|gonna)\s+(?:work\s+on|do|use|try)\s+(?:this|it|that)\s+later)\b',
            r'\b(?:save\s+(?:for|this\s+for)\s+later|bookmark\s+(?:this|it))\b',
        ]
        for pattern in deferral_patterns:
            if re.search(pattern, lower):
                return 'deferral'

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Suggestion emission
    # ──────────────────────────────────────────────────────────────────────────

    def emit_save_card(self, thread_id: str, topic: str, content_type: str) -> None:
        """Emit a save suggestion card via DocumentCardService."""
        try:
            from services.document_card_service import DocumentCardService

            card_svc = DocumentCardService()
            card_svc.emit_save_suggestion_card(topic, content_type, thread_id)
            logger.info(f"{LOG_PREFIX} Emitted save suggestion card for {content_type}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_save_card failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Document creation (called when user accepts)
    # ──────────────────────────────────────────────────────────────────────────

    def create_document_from_conversation(
        self,
        thread_id: str,
        topic: str,
        content_type: str,
    ) -> Optional[str]:
        """
        Create a document from conversation content:
        1. Get conversation anchored at the flagged exchange
        2. Check for duplicates (skip if identical in last hour)
        3. Synthesize clean markdown via LLM
        4. Create document via DocumentService
        5. Enqueue processing
        6. Clear saveable flag
        Returns doc_id or None.
        """
        try:
            # 1. Get conversation window
            conversation = self._get_conversation_window(thread_id)
            if not conversation:
                logger.warning(f"{LOG_PREFIX} No conversation content for thread {thread_id}")
                return None

            # 2. Duplicate check
            conv_hash = hashlib.sha256(conversation.encode()).hexdigest()
            if self._is_duplicate(conv_hash):
                logger.info(f"{LOG_PREFIX} Duplicate detected, skipping")
                self.clear_flag(thread_id)
                return None

            # 3. Synthesize document via LLM
            doc_text = self._synthesize_document(conversation, content_type)
            if not doc_text:
                logger.warning(f"{LOG_PREFIX} LLM synthesis returned empty")
                return None

            # 4. Create document
            display_type = content_type.replace('_', ' ').title()
            safe_topic = re.sub(r'[^\w\s-]', '', (topic or 'document')).strip()[:50]
            doc_name = f"{safe_topic} - {display_type}.md"

            from services.database_service import get_shared_db_service
            from services.document_service import DocumentService

            doc_svc = DocumentService(get_shared_db_service())
            doc_id = doc_svc.create_document_from_text(doc_name, doc_text, 'conversation')

            # 5. Enqueue processing
            try:
                from services import PromptQueue
                from workers.document_worker import process_document_job
                queue = PromptQueue(queue_name="document-queue", worker_func=process_document_job)
                queue.enqueue({'doc_id': doc_id})
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to enqueue processing: {e}")

            # 6. Clear flag + set cooldown
            self.clear_flag(thread_id)
            self._set_cooldown(thread_id)

            logger.info(f"{LOG_PREFIX} Created document {doc_id} from conversation")
            return doc_id

        except Exception as e:
            logger.error(f"{LOG_PREFIX} create_document_from_conversation failed: {e}",
                         exc_info=True)
            return None

    def record_rejection(self, thread_id: str, topic: str) -> None:
        """Record a save suggestion rejection for rate limiting."""
        try:
            redis = self._get_redis()
            # Set cooldown (prevent re-suggesting for 1 hour)
            cooldown_key = COOLDOWN_KEY.format(thread_id=thread_id)
            redis.setex(cooldown_key, COOLDOWN_TTL, '1')
            # Set topic-level rejection (prevent for 7 days on same topic)
            if topic:
                reject_key = REJECT_KEY.format(
                    thread_id=thread_id,
                    topic=topic,
                )
                redis.setex(reject_key, REJECT_TTL, '1')
            logger.info(f"{LOG_PREFIX} Recorded rejection for thread {thread_id}, topic {topic}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} record_rejection failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_redis(self):
        from services.redis_client_service import RedisClientService
        return RedisClientService.create_connection()

    def _get_conversation_window(self, thread_id: str) -> Optional[str]:
        """Get recent conversation turns formatted for LLM synthesis."""
        try:
            from services.working_memory_service import WorkingMemoryService

            wm = WorkingMemoryService()
            turns = wm.get_recent_turns(thread_id, n=10)
            if not turns:
                return None

            lines = []
            for turn in turns:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                if role == 'user':
                    lines.append(f"User: {content}")
                elif role == 'assistant':
                    lines.append(f"Assistant: {content}")

            return '\n\n'.join(lines)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _get_conversation_window failed: {e}")
            return None

    def _synthesize_document(self, conversation: str, content_type: str) -> Optional[str]:
        """Use LLM to synthesize clean markdown from conversation."""
        try:
            from services.config_service import ConfigService
            from services.llm_service import create_llm_service

            config = ConfigService.resolve_agent_config('document-synthesis')
            # Override format for plain text output (not JSON)
            config['format'] = ''
            llm = create_llm_service(config)

            # Load prompt template
            prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'prompts', 'document-synthesis-from-conversation.md',
            )
            with open(prompt_path, 'r') as f:
                template = f.read()

            prompt = template.replace(
                '{{content_type}}', content_type.replace('_', ' ')
            ).replace(
                '{{conversation}}', conversation
            )

            response = llm.send_message(
                "You are a document extraction assistant.",
                prompt,
            )
            text = response.text.strip() if response and response.text else None

            # Strip think tags if present
            if text:
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                if '<think>' in text:
                    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()

            return text
        except Exception as e:
            logger.error(f"{LOG_PREFIX} _synthesize_document failed: {e}")
            return None

    def _is_duplicate(self, conv_hash: str) -> bool:
        """Check if we've already created a document from this exact conversation."""
        try:
            redis = self._get_redis()
            dup_key = f"save_suggest:hash:{conv_hash}"
            if redis.exists(dup_key):
                return True
            # Mark as seen for 1 hour
            redis.setex(dup_key, 3600, '1')
            return False
        except Exception:
            return False

    def _set_cooldown(self, thread_id: str) -> None:
        """Set cooldown to prevent re-suggesting."""
        try:
            redis = self._get_redis()
            key = COOLDOWN_KEY.format(thread_id=thread_id)
            redis.setex(key, COOLDOWN_TTL, '1')
        except Exception:
            pass
