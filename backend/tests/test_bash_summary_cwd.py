"""Feature test for BashAbility's mp-gated summary enrichment (TKT-837).

bash is the canonical example of a getter that enriches on a live request:
``get_summary()`` appends the working directory when bound to a real processor
(``self.mp is not None``) so the model knows where commands run, but returns the
bare base text at ``self.mp is None`` (build / search-index time) so
``abilities.sqlite`` + the SHA drift map stay machine-independent.

Real BashAbility, no mocks — the only injected value is a minimal real
MP-shaped context carrying a real channel config (what ``get_summary`` reads
nothing off, but mirrors the live binding the dispatcher performs).
"""

from pathlib import Path

import pytest

from abilities.bash import BashAbility
from configs.channels import UserConfig

pytestmark = pytest.mark.unit


class _Mp:
    """Minimal real MP-shaped context — bash's get_summary only checks that
    ``self.mp is not None``; a live processor carries a real config."""

    def __init__(self, config):
        self.config = config


def test_summary_is_bare_base_text_at_build_time():
    """mp=None (search-index build / introspection) → no cwd, deterministic text.

    This is the invariant that keeps the built abilities.sqlite + abilities_sha
    identical across machines regardless of where the build runs.
    """
    summary = BashAbility().get_summary()
    assert "Working directory" not in summary
    assert summary == BashAbility._SUMMARY


def test_summary_appends_cwd_on_a_live_request():
    """Bound to a live processor → the working directory is appended so the model
    knows where commands run. The base text is preserved verbatim as the prefix."""
    bash = BashAbility(mp=_Mp(UserConfig({})))
    summary = bash.get_summary()

    assert summary.startswith(BashAbility._SUMMARY)
    assert f"Working directory: {Path.home()}" in summary
    assert "Use absolute paths or cd to operate elsewhere." in summary


def test_descriptor_description_reflects_the_live_summary():
    """The single get_input_schema assembler uses get_summary() for the
    descriptor's 'description', so the cwd enrichment flows through to what the
    model actually sees — and act_summary is still injected as required."""
    descriptor = BashAbility(mp=_Mp(UserConfig({}))).get_input_schema()

    assert f"Working directory: {Path.home()}" in descriptor["description"]
    assert "act_summary" in descriptor["input_schema"]["properties"]
    assert "act_summary" in descriptor["input_schema"]["required"]
