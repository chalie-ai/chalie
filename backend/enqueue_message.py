#!/usr/bin/env python3
"""Simple script to enqueue messages to the prompt queue."""
import sys
from rq import Queue
from redis import Redis

def enqueue_message(message, metadata=None):
    """Enqueue a message to the prompt-queue using RQ."""
    # Connect to Redis
    redis_conn = Redis(host='grck.lan', port=6379, decode_responses=False)

    # Create RQ queue
    queue = Queue(name='prompt-queue', connection=redis_conn)

    # Enqueue the job - RQ expects the function path as a string
    # The worker will call: digest_worker(message, metadata=metadata)
    job = queue.enqueue(
        'workers.digest_worker.digest_worker',
        message,
        metadata=metadata or {}
    )

    print(f"✓ Enqueued job {job.id} to prompt-queue")
    print(f"  Message: {message[:50]}...")
    return job.id

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 enqueue_message.py '<message>'")
        sys.exit(1)

    message = sys.argv[1]
    metadata = {'source': 'cli_audit', 'uuid': 'claude-test'}

    enqueue_message(message, metadata)
