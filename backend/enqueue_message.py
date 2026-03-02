#!/usr/bin/env python3
"""Simple script to enqueue messages to the prompt queue."""
import sys
import uuid


def enqueue_message(message, metadata=None):
    """Enqueue a message via the digest worker (thread-based)."""
    from services.prompt_queue import PromptQueue
    from workers.digest_worker import digest_worker

    metadata = metadata or {}
    if 'uuid' not in metadata:
        metadata['uuid'] = str(uuid.uuid4())

    queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
    queue.enqueue(message, metadata=metadata)

    print(f"Enqueued message to prompt-queue")
    print(f"  Message: {message[:50]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 enqueue_message.py '<message>'")
        sys.exit(1)

    message = sys.argv[1]
    metadata = {'source': 'cli_audit', 'uuid': str(uuid.uuid4())}

    enqueue_message(message, metadata)
