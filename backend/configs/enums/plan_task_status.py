# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PlanTaskStatus — where one task of a message's plan stands.

The member VALUE is the literal stored in ``plan_tasks.status`` (the schema's
CHECK constraint lists the same four), the word the model passes to
``plan(action="update")``, and the state the chat's plan card draws.
"""

from __future__ import annotations

from enum import Enum


class PlanTaskStatus(str, Enum):
    TODO        = "todo"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    CANCELLED   = "cancelled"
