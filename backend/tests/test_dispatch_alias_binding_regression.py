# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

from abilities._registry import AbilityRegistry

pytestmark = pytest.mark.unit


def test_real_tool_resolves_from_registry() -> None:
    names = {a.NAME for a in AbilityRegistry.all()}
    assert "memory" in names
    assert "find_tools" in names
