# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Controllers — the orchestration layer.

Holds the single MP-spine orchestrator (:class:`controllers.message_processor.MessageProcessor`,
§1.1): the one class that drives an LLM turn end-to-end. Every DB-stored,
LLM-tracking object it touches is reached through a coordinating service off
``self`` (Rules 3–9); this package imports the service layer, never the reverse
(services type-reference the controller under ``TYPE_CHECKING`` only).
"""
