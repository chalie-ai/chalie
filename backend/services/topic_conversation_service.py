import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class TopicConversationService:
    """Service for managing topic-based conversation files with confidence tracking."""

    def __init__(self, conversations_dir: Optional[Path] = None):
        """
        Initialize the service with a conversations directory.

        Args:
            conversations_dir: Path to conversations directory. Defaults to ../conversations
        """
        if conversations_dir is None:
            conversations_dir = Path(__file__).resolve().parent.parent / "conversations"

        self.conversations_dir = conversations_dir
        self.conversations_dir.mkdir(exist_ok=True)

    def handle_classification(
        self,
        text: str,
        classification: Dict[str, Any],
        classification_time: float,
        min_trusted_confidence: int = 7
    ) -> Tuple[str, str]:
        """
        Handle classifier output and update conversation files accordingly.

        Args:
            text: The original prompt text
            classification: Classifier response dict with topic, confidence, similar_topic, topic_update
            classification_time: Time taken for classification
            min_trusted_confidence: Minimum confidence to rename files (default: 7)

        Returns:
            Tuple[str, str]: The topic name and exchange ID for this prompt
        """
        topic = classification.get('topic', '')
        confidence = classification.get('confidence', 0)
        similar_topic = classification.get('similar_topic', '')
        topic_update = classification.get('topic_update', '')

        # Create prompt data with unique ID
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        prompt_data = {
            "id": str(uuid.uuid4()),
            "message": text,
            "time": timestamp,
            "classification_time": classification_time
        }

        # Scenario 1: New topic (no similar_topic)
        # Use topic_update if provided (better name), otherwise use topic
        if not similar_topic:
            topic_name = topic_update if topic_update else topic

            # Validate topic name is not empty
            if not topic_name or not topic_name.strip():
                topic_name = "unclassified"
                print(f"log [topic_conversation]: WARNING - Classifier returned empty topic, using 'unclassified'")

            exchange_id = self._add_prompt_to_conversation(topic_name, prompt_data, confidence)
            print(f"log [topic_conversation]: Created new conversation '{topic_name}'")
            return topic_name, exchange_id

        # Scenario 2 & 3: Update existing topic
        if similar_topic:
            # Check if we should rename (topic_update + high confidence)
            should_rename = (
                topic_update
                and confidence >= min_trusted_confidence
            )

            if should_rename:
                exchange_id = self._rename_and_update_conversation(
                    similar_topic,
                    topic_update,
                    confidence,
                    prompt_data
                )
                print(f"log [topic_conversation]: Renamed conversation from '{similar_topic}' to '{topic_update}'")
                return topic_update, exchange_id
            else:
                # Just update existing conversation
                exchange_id = self._add_prompt_to_conversation(similar_topic, prompt_data, confidence)
                print(f"log [topic_conversation]: Updated conversation '{similar_topic}'")
                return similar_topic, exchange_id

        # Fallback (should be unreachable)
        fallback_topic = topic_update if topic_update else topic

        # Validate topic name is not empty
        if not fallback_topic or not fallback_topic.strip():
            fallback_topic = "unclassified"
            print(f"log [topic_conversation]: WARNING - Classifier returned empty topic in fallback, using 'unclassified'")

        print(f"log [topic_conversation]: No action taken, using topic '{fallback_topic}'")
        return fallback_topic, prompt_data['id']

    def _add_prompt_to_conversation(
        self,
        topic: str,
        prompt_data: Dict[str, Any],
        new_confidence: int
    ) -> str:
        """
        Add a prompt to a conversation, creating the file if needed.

        Args:
            topic: The topic name (will be normalized)
            prompt_data: Prompt data dict with id, message, time, classification_time
            new_confidence: New confidence score to apply

        Returns:
            str: The exchange ID
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            # Create new conversation
            conversation_data = {
                "topic": normalized_topic,
                "confidence": new_confidence,
                "status": "pending",
                "exchanges": [{"prompt": prompt_data}]
            }
        else:
            # Load and update existing conversation
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            # Update confidence
            current_confidence = conversation_data.get('confidence', 0)
            conversation_data['confidence'] = self._calculate_new_confidence(
                current_confidence,
                new_confidence
            )

            # Append new exchange
            conversation_data['exchanges'].append({"prompt": prompt_data})

        # Write to file
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        return prompt_data['id']

    def _rename_and_update_conversation(
        self,
        old_topic: str,
        new_topic: str,
        new_confidence: int,
        prompt_data: Dict[str, Any]
    ) -> str:
        """
        Rename conversation file and update with new topic name and confidence.

        Returns:
            str: The exchange ID
        """
        old_normalized = self._normalize_topic(old_topic)
        new_normalized = self._normalize_topic(new_topic)

        old_file = self.conversations_dir / f"{old_normalized}.json"
        new_file = self.conversations_dir / f"{new_normalized}.json"

        if not old_file.exists():
            # If old file doesn't exist, create new one
            return self._add_prompt_to_conversation(new_topic, prompt_data, new_confidence)

        # Read existing data
        with open(old_file, 'r') as f:
            conversation_data = json.load(f)

        # Update topic name
        conversation_data['topic'] = new_normalized

        # Update confidence score
        current_confidence = conversation_data.get('confidence', 0)
        updated_confidence = self._calculate_new_confidence(
            current_confidence,
            new_confidence
        )
        conversation_data['confidence'] = updated_confidence

        # Append new exchange
        conversation_data['exchanges'].append({"prompt": prompt_data})

        # Write to new file
        with open(new_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)

        # Remove old file
        old_file.unlink()

        return prompt_data['id']

    def _calculate_new_confidence(
        self,
        current_confidence: int,
        new_confidence: int
    ) -> int:
        """
        Calculate confidence using bounded reinforcement with decay.

        Args:
            current_confidence: Current confidence in file
            new_confidence: New confidence from classifier

        Returns:
            Updated confidence (clamped to 0-10)
        """
        alpha = 0.6  # memory
        beta = 0.4   # new signal

        adjusted = (current_confidence * alpha) + (new_confidence * beta)

        return max(0, min(10, round(adjusted)))

    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic name to lowercase with hyphens."""
        return topic.lower().replace(' ', '-')

    def get_existing_topics(
        self,
        message_confidence_bonus: float = 0.5,
        recent_message_bonus: float = 2.0,
        recency_window: int = 5
    ) -> list:
        """
        Get all existing conversation topics with recency bonus applied.
        Only returns topics with confidence > mean.

        Extracts ALL messages from ALL conversations, sorts by timestamp,
        and applies recency bonus to conversations containing the top 5 most recent messages.

        Args:
            message_confidence_bonus: Bonus per message in recency window (default 0.5)
            recent_message_bonus: Bonus for most recent message (default 2.0)
            recency_window: Number of recent messages to apply bonus (default 5)

        Returns:
            List of topic names with above-average confidence
        """
        if not self.conversations_dir.exists():
            return []

        # Load all conversations
        conversations = {}
        all_messages = []

        for file in self.conversations_dir.glob("*.json"):
            with open(file, 'r') as f:
                conversation = json.load(f)
                topic = conversation.get('topic', '')
                base_confidence = conversation.get('confidence', 0)
                exchanges = conversation.get('exchanges', [])

                if not topic:
                    continue

                conversations[topic] = {
                    'confidence': base_confidence,
                    'recency_bonus': 0.0
                }

                # Extract all messages with their topic and timestamp
                for exchange in exchanges:
                    prompt = exchange.get('prompt', {})
                    timestamp = prompt.get('time', '')
                    if timestamp:
                        all_messages.append({
                            'topic': topic,
                            'time': timestamp
                        })

        if not conversations:
            return []

        # Sort all messages by timestamp (most recent first)
        all_messages.sort(key=lambda x: x['time'], reverse=True)

        # Apply recency bonus to top recency_window messages
        for i, message in enumerate(all_messages[:recency_window]):
            topic = message['topic']
            if i == 0:
                # Most recent message gets the full bonus
                conversations[topic]['recency_bonus'] += recent_message_bonus
            else:
                # Messages 2-5 get the standard bonus
                conversations[topic]['recency_bonus'] += message_confidence_bonus

        # Calculate adjusted confidence for each conversation
        conversations_with_confidence = []
        for topic, data in conversations.items():
            adjusted_confidence = data['confidence'] + data['recency_bonus']
            conversations_with_confidence.append({
                'topic': topic,
                'confidence': adjusted_confidence
            })

        # Calculate mean confidence
        total_confidence = sum(c['confidence'] for c in conversations_with_confidence)
        mean_confidence = total_confidence / len(conversations_with_confidence)

        # Filter conversations where confidence > mean
        filtered_topics = [
            c['topic']
            for c in conversations_with_confidence
            if c['confidence'] > mean_confidence
        ]

        # Ensure at least 2 conversations are returned
        if len(filtered_topics) < 2:
            # Sort by confidence and take top 2 (or all if fewer than 2 total)
            sorted_conversations = sorted(
                conversations_with_confidence,
                key=lambda x: x['confidence'],
                reverse=True
            )
            filtered_topics = [c['topic'] for c in sorted_conversations[:2]]

        return filtered_topics

    def get_conversation_history(self, topic: str) -> list:
        """
        Retrieve all exchanges for a given topic.

        Args:
            topic: The topic name to retrieve history for

        Returns:
            list: All exchanges with prompt and response data, or empty list if not found
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            return []

        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)
            return conversation_data.get('exchanges', [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read conversation file for '{topic}': {e}")
            return []

    def add_response(
        self,
        topic: str,
        response_message: str,
        generation_time: float
    ) -> None:
        """
        Add a response to the most recent exchange in a conversation.

        Args:
            topic: The topic name
            response_message: The generated response text
            generation_time: Time taken to generate the response

        Returns:
            None
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            print(f"Warning: Cannot add response - conversation file '{normalized_topic}' does not exist")
            return

        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            # Add response to the last exchange
            if conversation_data.get('exchanges'):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                response_data = {
                    "message": response_message,
                    "time": timestamp,
                    "generation_time": generation_time
                }
                conversation_data['exchanges'][-1]['response'] = response_data

                with open(conversation_file, 'w') as f:
                    json.dump(conversation_data, f, indent=2)
            else:
                print(f"Warning: No exchanges found in conversation '{normalized_topic}'")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Could not update conversation file for '{topic}': {e}")

    def add_response_error(
        self,
        topic: str,
        error_message: str
    ) -> None:
        """
        Record a response generation error in the conversation.

        Args:
            topic: The topic name
            error_message: The error message to record

        Returns:
            None
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            print(f"Warning: Cannot add error - conversation file '{normalized_topic}' does not exist")
            return

        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            # Add error to the last exchange
            if conversation_data.get('exchanges'):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                error_data = {
                    "error": error_message,
                    "time": timestamp
                }
                conversation_data['exchanges'][-1]['response'] = error_data

                with open(conversation_file, 'w') as f:
                    json.dump(conversation_data, f, indent=2)
            else:
                print(f"Warning: No exchanges found in conversation '{normalized_topic}'")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Could not update conversation file for '{topic}': {e}")

    def add_steps_to_exchange(self, topic: str, next_actions: list) -> None:
        """
        Add steps derived from next_actions to the most recent exchange.

        Args:
            topic: The topic name
            next_actions: List of action dicts from FrontalCortexService
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            return

        with open(conversation_file, 'r') as f:
            conversation_data = json.load(f)

        if conversation_data.get('exchanges'):
            # Transform next_actions into steps with status
            steps = []
            for action in next_actions:
                step = {
                    "type": action.get("type", "task"),
                    "description": action.get("description", ""),
                    "status": "pending"
                }
                # Include optional fields
                if "when" in action:
                    step["when"] = action["when"]
                if "query" in action:
                    step["query"] = action["query"]
                steps.append(step)

            conversation_data['exchanges'][-1]['steps'] = steps

            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)

    def update_topic_status(self, topic: str) -> None:
        """
        Calculate and update topic status based on exchange steps.

        Status logic:
        - "completed": all exchanges have no steps OR all steps are completed
        - "scheduled": has scheduled steps and other pending steps
        - "in progress": has in-progress steps
        - "pending": default
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            return

        with open(conversation_file, 'r') as f:
            conversation_data = json.load(f)

        exchanges = conversation_data.get('exchanges', [])
        if not exchanges:
            conversation_data['status'] = 'pending'
        else:
            # Collect all steps from all exchanges
            all_steps = []
            for exchange in exchanges:
                steps = exchange.get('steps', [])
                all_steps.extend(steps)

            # Determine status
            if not all_steps:
                new_status = 'completed'
            else:
                has_in_progress = any(s.get('status') == 'in progress' for s in all_steps)
                has_scheduled = any(s.get('status') == 'scheduled' for s in all_steps)
                all_completed = all(s.get('status') == 'completed' for s in all_steps)

                if all_completed:
                    new_status = 'completed'
                elif has_in_progress:
                    new_status = 'in progress'
                elif has_scheduled:
                    new_status = 'scheduled'
                else:
                    new_status = 'pending'

            conversation_data['status'] = new_status

        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)

    def add_memory_chunk(self, topic: str, exchange_id: str, memory_chunk: dict) -> None:
        """
        Add a memory chunk to a specific exchange in a conversation.

        Args:
            topic: The topic name
            exchange_id: The unique ID of the exchange to update
            memory_chunk: The structured memory data (dict with scope, emotion, gists)
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            print(f"Warning: Cannot add memory chunk - conversation file '{normalized_topic}' does not exist")
            return

        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            # Find the exchange by ID and add memory_chunk
            exchanges = conversation_data.get('exchanges', [])
            for exchange in exchanges:
                if exchange.get('prompt', {}).get('id') == exchange_id:
                    exchange['memory_chunk'] = memory_chunk

                    with open(conversation_file, 'w') as f:
                        json.dump(conversation_data, f, indent=2)
                    return

            print(f"Warning: Exchange ID '{exchange_id}' not found in conversation '{normalized_topic}'")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Could not update conversation file for '{topic}': {e}")

    def remove_exchanges(self, topic: str, exchange_ids: list) -> None:
        """
        Remove consolidated exchanges from a conversation file.

        Args:
            topic: The topic name
            exchange_ids: List of exchange IDs to remove
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            print(f"Warning: Cannot remove exchanges - conversation file '{normalized_topic}' does not exist")
            return

        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            # Filter out exchanges with matching IDs
            exchanges = conversation_data.get('exchanges', [])
            filtered_exchanges = [
                e for e in exchanges
                if e.get('prompt', {}).get('id') not in exchange_ids
            ]

            conversation_data['exchanges'] = filtered_exchanges

            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)

            print(f"log [topic_conversation]: Removed {len(exchanges) - len(filtered_exchanges)} exchanges from '{topic}'")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Could not remove exchanges from '{topic}': {e}")

    def delete_topic_if_empty(self, topic: str) -> None:
        """
        Delete topic conversation file if it has no exchanges.
        This allows the system to "forget" short-term memory when topic switches.

        Args:
            topic: The topic name
        """
        normalized_topic = self._normalize_topic(topic)
        conversation_file = self.conversations_dir / f"{normalized_topic}.json"

        if not conversation_file.exists():
            return

        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            exchanges = conversation_data.get('exchanges', [])

            if not exchanges:
                conversation_file.unlink()
                print(f"log [topic_conversation]: Deleted empty conversation file for '{topic}'")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Could not check/delete conversation file for '{topic}': {e}")
