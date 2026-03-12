# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Backward-compatible import alias.

The cognitive drift engine has been renamed to ReasoningLoopService
(reasoning_loop_service.py). This module re-exports all public names
so that existing ``from services.cognitive_drift_engine import ...``
statements continue to work.
"""

from services.reasoning_loop_service import (  # noqa: F401
    ReasoningSignal,
    emit_reasoning_signal,
    ReasoningLoopService as CognitiveDriftEngine,
    reasoning_loop_worker as cognitive_drift_worker,
)

__all__ = [
    'ReasoningSignal',
    'emit_reasoning_signal',
    'CognitiveDriftEngine',
    'cognitive_drift_worker',
]
