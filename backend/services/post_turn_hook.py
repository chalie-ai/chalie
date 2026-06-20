# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PostTurnHook — one independent unit of after-turn work.

After the turn's final assistant row is persisted,
``MessageProcessor._end_turn`` runs every hook in ``mp.config.post_turn_hooks``. Each hook is a single-responsibility,
self-contained object: a config composes the ones it needs, and cross-cutting
after-turn behaviours (proactive suggestion, pattern decay, disclosure-to-human,
summary persistence) become reusable units instead of one opaque callable field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only: hooks receive the turn's processor but never import it at
    # runtime (the processor imports this module's consumers, not vice versa).
    from services.message_processor import MessageProcessor


class PostTurnHook(ABC):
    """One independent, self-contained unit of after-turn work.

    INDEPENDENCE CONTRACT (load-bearing — do not weaken):
      - Hooks are mutually INDEPENDENT. A hook MUST NOT depend on, observe, or
        order itself relative to any other hook. There is NO defined execution
        order, and none may ever be assumed.
      - The framework MAY run hooks in any order, and MAY in the future run them
        concurrently / fully async from one another. A hook must therefore be
        self-contained: it owns its own context (copy contextvars if it touches
        request-scoped state), holds no reference to sibling hooks, and shares no
        mutable state with them.
      - FAILURE ISOLATION: a hook that raises MUST NOT affect any sibling hook.
        The invoker isolates each call (log + continue). One hook failing is a
        non-event for the others.
    """

    @abstractmethod
    def run(self, mp: "MessageProcessor", result_text: str) -> None:
        """Called once per turn after the assistant row is persisted.
        Any result is a side effect (DB write, dispatched message, log).
        """
