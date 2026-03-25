"""Headless browser tool — JS rendering, screenshots, interaction, page monitoring.

Optional dependency: requires `playwright` with Chromium installed.
Auto-detected at import time. Tool excluded from registry if unavailable.
"""

try:
    import playwright  # noqa: F401
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
