#!/usr/bin/env python3
"""
Test that empty topic names are handled correctly.
"""
import sys
sys.path.insert(0, 'src')

from services.topic_conversation_service import TopicConversationService
from services.logger_service import LoggerService

logger = LoggerService.get_logger(__name__)

# Test case: Empty topic from classifier
logger.info("Test: Empty topic handling...")
service = TopicConversationService()

# Simulate classifier returning empty topic
classification = {
    'topic': '',
    'confidence': 3,
    'similar_topic': '',
    'topic_update': ''
}

topic, exchange_id = service.handle_classification(
    text="hey",
    classification=classification,
    classification_time=0.5
)

assert topic == "unclassified", f"Expected 'unclassified', got '{topic}'"
assert exchange_id is not None, "Exchange ID should not be None"
logger.info(f"✓ Empty topic handled correctly: '{topic}'")

# Verify file was created with proper name
import os
from pathlib import Path
conversations_dir = Path(__file__).resolve().parent.parent / "conversations"
expected_file = conversations_dir / "unclassified.json"
assert expected_file.exists(), f"Expected file {expected_file} to exist"
logger.info(f"✓ File created correctly: {expected_file}")

# Clean up
if expected_file.exists():
    expected_file.unlink()
    logger.info(f"✓ Cleaned up test file")

logger.info("\n✅ All empty topic validation tests passed")
