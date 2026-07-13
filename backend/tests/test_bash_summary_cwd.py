"""Feature test for BashAbility's mp-gated summary enrichment.

Invariants: abilities.sqlite + SHA drift map are identical across machines at
build time (self.mp is None); cwd enriches only on a real processor.

Real BashAbility, no mocks — the only injected value is a minimal real
MP-shaped context carrying a real channel config.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from abilities.bash import BashAbility
from configs.channels import UserConfig

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


class _Mp:
    def __init__(self, config: object) -> None:
        self.config = config


def test_summary_is_bare_base_text_at_build_time() -> None:
    summary = BashAbility().get_summary()
    assert "Working directory" not in summary
    assert summary == BashAbility._SUMMARY


def test_summary_appends_cwd_on_a_live_request() -> None:
    """Bound to a live processor → the working directory is appended so the model
    knows where commands run. The base text is preserved verbatim as the prefix."""
    bash = BashAbility(mp=cast("MessageProcessor", _Mp(UserConfig({}))))
    summary = bash.get_summary()

    assert summary.startswith(BashAbility._SUMMARY)
    assert f"Working directory: {Path.home()}" in summary
    assert "Use absolute paths or cd to operate elsewhere." in summary


def test_descriptor_description_reflects_the_live_summary() -> None:
    """The single get_input_schema assembler uses get_summary() for the
    descriptor's 'description', so the cwd enrichment flows through to what the
    model actually sees — and act_summary is still injected as required."""
    descriptor = BashAbility(mp=cast("MessageProcessor", _Mp(UserConfig({})))).get_input_schema()

    assert f"Working directory: {Path.home()}" in cast(str, descriptor["description"])
    assert "act_summary" in cast(dict[str, object], cast(dict[str, object], descriptor["input_schema"])["properties"])
    assert "act_summary" in cast(list[object], cast(dict[str, object], descriptor["input_schema"])["required"])
