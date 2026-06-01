# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
configs — per-channel ProcessorConfig constants and factory functions.

Each channel's behavioural surface is expressed as a frozen ProcessorConfig
instance.  Simple channels expose module-level constants; channels that require
per-instance arguments expose factory functions.

Spec: ACT Loop Orchestrator Refactor §3.
"""
