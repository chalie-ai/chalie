# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Post-Exchange Hooks — formerly housed adaptive-layer fork detection and signal
writes.  AdaptiveLayerService was removed in v0.5.0 §4.3.  This module is kept
as a shell so existing importability checks in tests/test_webhook_rip_removed.py
continue to pass without error.
"""

import logging

logger = logging.getLogger(__name__)
