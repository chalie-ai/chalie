# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Prompt Assembly Contract — shared interface for prompt assembly services.

Both SystemPromptAssemblyService and UserPromptAssemblyService implement
this contract. The `to_provider()` method returns the final string that
gets sent to the LLM provider in either the system-prompt or user-prompt field.
"""

from abc import ABC, abstractmethod


class PromptAssemblyContract(ABC):
    """Interface that all prompt assembly services must implement.

    The single method `to_provider()` returns a ready-to-send string.
    Callers never need to know how the prompt was assembled — they just
    call `to_provider()` and pass the result to the LLM provider.
    """

    @abstractmethod
    def to_provider(self) -> str:
        """Return the assembled prompt string ready for the LLM provider.

        For SystemPromptAssemblyService: the stable system prompt.
        For UserPromptAssemblyService: the per-turn user message.
        """
        ...
