"""Whether iMessage can work where Chalie runs — decided here, once.

Chalie sends iMessages by driving the Messages app, which exists only on a
Mac. The capability, its setup, its tool and whether Chalie is told about the
tool all read these two names, so they always agree.

This module imports nothing from Chalie, so the ability can read it while the
registry loads without pulling in the capability itself.
"""

import sys

AVAILABLE: bool = sys.platform == "darwin"

APPLE_ONLY = "iMessage is Apple-only: it works only when Chalie runs on a Mac, through the Messages app."
