"""In-development feature gating.

A capability that is not yet ready for general release is enabled only in
processes where ``CHALIE_INTERNAL_DEV`` is set to ``"1"``. Released images never
set it, so every gated feature stays hidden by default — no per-feature config,
one switch.
"""

import os


_INTERNAL_DEV_ENV = "CHALIE_INTERNAL_DEV"


class FeatureFlags:
    """In-development feature gating."""

    @staticmethod
    def internal_dev_enabled() -> bool:
        """True when in-development features are enabled for this process."""
        return os.environ.get(_INTERNAL_DEV_ENV, "") == "1"
