# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared type aliases and primitives for the Chalie backend.

This module is the canonical home for recurring shape aliases, TypedDicts,
and Protocols that appear across multiple packages. As packages are migrated
to mypy ``--strict`` (see ``docs/typing-ratchet.md``), they should import
aliases from here rather than re-declaring ``dict[str, Any]`` inline.

Aliases defined here:

- :data:`JSONDict` — ``dict[str, Any]``: an arbitrary JSON-decoded object.
  Used throughout services for raw database row dicts, API payloads, and
  provider config maps.
- :data:`JSONList` — ``list[dict[str, Any]]``: a list of JSON-decoded objects.
  Returned by service methods that query multiple rows (e.g. ``get_all_*``).
"""

from typing import Any

# ---------------------------------------------------------------------------
# Basic JSON shape aliases
# ---------------------------------------------------------------------------

JSONDict = dict[str, Any]
"""Arbitrary JSON-decoded mapping (``dict[str, Any]``).

Prefer this alias over repeating ``dict[str, Any]`` across service and
ability signatures. When a package is migrated to strict and a more precise
TypedDict is available, switch to that — ``JSONDict`` is an intentional
stepping stone, not a permanent solution.
"""

JSONList = list[dict[str, Any]]
"""List of JSON-decoded mappings (``list[dict[str, Any]]``).

Returned by service methods that load multiple rows from the database or
multiple items from an external API. Replace with a typed list once the
element TypedDict exists.
"""
