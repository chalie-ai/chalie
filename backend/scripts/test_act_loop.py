#!/usr/bin/env python3
"""
Quick sanity check for ACT loop functionality.
Run after implementation to verify basic operation.
"""
import sys
sys.path.insert(0, 'src')

from services.act_loop_service import ActLoopService
from services.act_dispatcher_service import ActDispatcherService
from services.logger_service import LoggerService

logger = LoggerService.get_logger(__name__)

# Test 1: ActLoopService basic operations
logger.info("Test 1: ActLoopService initialization...")
act_loop = ActLoopService(max_iterations=3, cumulative_timeout=60.0)
assert act_loop.can_continue() == True
logger.info("✓ Initialized correctly")

# Test 2: History formatting
logger.info("\nTest 2: History context formatting...")
act_loop.append_results([{
    'action_type': 'memory_query',
    'status': 'success',
    'result': 'Found 2 episodes',
    'execution_time': 0.5
}])
history = act_loop.get_history_context()
assert 'memory_query' in history
assert 'Found 2 episodes' in history
logger.info("✓ History formatting works")

# Test 3: Iteration limit
logger.info("\nTest 3: Iteration limit enforcement...")
act_loop.iterations_remaining = 0
assert act_loop.can_continue() == False
logger.info("✓ Iteration limit enforced")

# Test 4: ActDispatcherService action handlers exist
logger.info("\nTest 4: Action handlers registered...")
dispatcher = ActDispatcherService()
assert 'memory_query' in dispatcher.handlers
assert 'memory_write' in dispatcher.handlers
assert 'world_state_read' in dispatcher.handlers
assert 'internal_reasoning' in dispatcher.handlers
logger.info("✓ All action handlers registered")

logger.info("\n✅ All sanity checks passed")
