"""
Test script for clearing database tables.
"""

import logging
from time import sleep
import sys
sys.path.insert(0, 'src')

from services.database_service import DatabaseService
from services.config_service import ConfigService
from services.prompt_queue import PromptQueue
from workers.digest_worker import digest_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_database():
    """
    Clear all cognitive AI tables from the database.
    Truncates all 7 tables using CASCADE option.
    """
    # Get database configuration
    db_config = ConfigService.get_agent_config("episodic-memory")
    connections_config = ConfigService.connections()
    
    # Merge configs
    merged_config = {
        'host': connections_config['postgresql']['host'],
        'port': connections_config['postgresql'].get('port', 5432),
        'database': connections_config['postgresql']['database'],
        'username': connections_config['postgresql'].get('username'),
        'password': connections_config['postgresql'].get('password'),
        'pool_size': db_config['database'].get('pool_size', 10),
        'max_overflow': db_config['database'].get('max_overflow', 20),
        'pool_timeout': db_config['database'].get('pool_timeout', 30)
    }
    
    # Initialize database service
    db_service = DatabaseService(merged_config)
    
    conn = None
    try:
        # Get connection from pool
        conn = db_service.get_connection()
        
        # Create cursor
        cursor = conn.cursor()
        
        # Truncate all tables with CASCADE
        tables = [
            'episodes',
            'cortex_iterations', 
            'semantic_concepts',
            'semantic_relationships',
            'semantic_schemas',
            'schema_version',
            'schema_migrations'
        ]
        
        for table in tables:
            cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
        
        # Commit transaction
        conn.commit()
        
        logger.info("Successfully truncated all cognitive AI tables")
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        logger.error(f"Failed to truncate tables: {e}")
        raise
    finally:
        # Release connection back to pool
        if conn:
            db_service.release_connection(conn)

def enqueue_prompts(prompt_list):
    """
    Enqueue prompts to the digest queue for async processing by workers.

    Args:
        prompt_list: List of prompt strings to enqueue

    Note: Make sure workers are running with: python3 src/consumer.py
    """
    # Create PromptQueue instance
    queue = PromptQueue(queue_name='prompt-queue', worker_func=digest_worker)

    total_prompts = len(prompt_list)
    counter = 0

    for i, prompt in enumerate(prompt_list, 1):
        try:
            # Enqueue with metadata
            queue.enqueue(prompt, metadata={
                'source': 'test_memory',
                'batch_index': i
            })
            logger.info(f"Enqueued {i} of {total_prompts} prompts: {prompt[:50]}...")

            # Rate limiting to avoid overwhelming the queue
            if 4 < counter < 6:
                logger.info("Rate limiting: sleeping 30 seconds...")
                sleep(30)

            if counter > 10:
                logger.info("Rate limiting: sleeping 5 minutes...")
                sleep(300)
                counter = 0

            counter += 1
        except Exception as e:
            logger.error(f"Failed to enqueue prompt {i}: {e}", exc_info=True)
            # Continue with remaining prompts
            continue

    logger.info(f"Finished enqueuing {total_prompts} prompts. Workers will process them asynchronously.")

def read_prompts_from_file(filepath):
    """
    Read prompts from a markdown file, one per line.
    Skips empty lines and lines starting with #.

    Args:
        filepath: Path to the prompts file

    Returns:
        List of prompt strings
    """
    prompts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    prompts.append(line)
        logger.info(f"Read {len(prompts)} prompts from {filepath}")
        return prompts
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error reading prompts file: {e}")
        return []

if __name__ == "__main__":
    # Clear database
    clear_database()

    # Read prompts from file and enqueue them
    prompts = read_prompts_from_file('prompts.md')
    if prompts:
        logger.info("Enqueuing prompts to worker queue...")
        logger.info("Make sure workers are running: python3 src/consumer.py")
        enqueue_prompts(prompts)
    else:
        logger.warning("No prompts found to process")