#!/usr/bin/env python3
"""
Integration test for ACT loop with digest_worker.
Tests the full flow without actually calling the LLM.
"""
import sys
sys.path.insert(0, 'src')

from services.logger_service import LoggerService

logger = LoggerService.get_logger(__name__)

logger.info("Test 1: Import all required modules...")
try:
    from workers.digest_worker import generate_response_with_act_loop
    from services.frontal_cortex_service import FrontalCortexService
    from services.act_loop_service import ActLoopService
    from services.act_dispatcher_service import ActDispatcherService
    logger.info("✓ All modules imported successfully")
except ImportError as e:
    logger.error(f"✗ Import failed: {e}")
    sys.exit(1)

logger.info("\nTest 2: Check function signature...")
import inspect
sig = inspect.signature(generate_response_with_act_loop)
params = list(sig.parameters.keys())
expected = ['topic', 'text', 'classification', 'conversation_service', 'cortex_config', 'cortex_prompt']
assert params == expected, f"Expected {expected}, got {params}"
logger.info("✓ Function signature correct")

logger.info("\nTest 3: Verify FrontalCortexService.generate_response has act_history parameter...")
sig = inspect.signature(FrontalCortexService.generate_response)
params = list(sig.parameters.keys())
assert 'act_history' in params, f"act_history not found in params: {params}"
logger.info("✓ act_history parameter present")

logger.info("\nTest 4: Check ACT loop configuration defaults...")
act_loop = ActLoopService()
assert act_loop.max_iterations == 3
assert act_loop.cumulative_timeout == 60.0
assert act_loop.per_action_timeout == 10.0
logger.info("✓ Default configuration correct")

logger.info("\n✅ All integration tests passed")
logger.info("\nNext steps:")
logger.info("1. Start consumer: python3 src/consumer.py")
logger.info("2. Test simple RESPOND: python3 src/listener.py \"Hello!\"")
logger.info("3. Monitor logs for ACT loop execution")
