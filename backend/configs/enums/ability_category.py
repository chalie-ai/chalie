# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AbilityCategory — the heading a tool is listed under in the ``find_tools`` menu.

Every DISCOVERABLE ability declares one (``Ability.CATEGORY``); the registry
raises at load time when it does not, so a new tool can never slip into the
roster uncategorised and render under no heading.

The member VALUE is the literal heading text the model reads, and **declaration
order here is render order** in the menu — there is no separate sort key to keep
in sync. Put the categories a chat turn reaches for most near the top.

Non-discoverable abilities (framework internals, delegate-pinned raw tools) leave
``CATEGORY`` as ``None``: they never appear in the menu, so a category would be a
value nothing reads.
"""

from __future__ import annotations

from enum import Enum


class AbilityCategory(str, Enum):
    FILE_OPERATIONS     = "File Operations"
    WEB                 = "Web"
    MEDIA               = "Media"
    INFORMATION         = "Information"
    PRODUCTIVITY        = "Productivity"
    SMART_HOME          = "Smart Home & Network"
    CONVERSATION        = "Conversation History"
    SYSTEM              = "System"
    EXTENSIONS          = "Extensions (MCP)"
    DELEGATE            = "Delegate (subagent)"
