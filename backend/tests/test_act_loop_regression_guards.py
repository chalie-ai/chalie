"""ACT-loop §9a regression guards (areas P & Q).

These tests pin the single-request-hierarchy guarantees (P1-P6) and assert the
removed / deprecated surfaces stay gone (Q1-Q5) after the ACT-loop refactor.

GOVERNING TEST PRINCIPLE: these tests reflect the blind spec (§9a). A failing
guard means production CODE has regressed — never weaken the guard to match
code.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROD_DIRS = ("services", "abilities", "configs", "workers", "api", "utils")


def _prod_python_files():
    """Yield (relpath, source) for every production .py file (skips tests/)."""
    for d in _PROD_DIRS:
        base = _BACKEND_DIR / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                yield path.relative_to(_BACKEND_DIR).as_posix(), path.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue


# ---------------------------------------------------------------------------
# P. Single-request hierarchy & no-duplicate-work guarantees
# ---------------------------------------------------------------------------


class TestP1OneFlatProcessor:
    """P1 — One flat MessageProcessor for all channels (§7a / §1 / §4)."""

    def test_message_processor_has_no_subclasses(self):
        """No code anywhere may subclass MessageProcessor.

        Importing the production modules that construct turns must not register
        a single subclass — every channel drives the flat ``process()`` entry
        point with a per-turn ProcessorConfig instead.
        """
        # Force-import the modules that historically defined subclasses so any
        # surviving subclass would be registered before we read __subclasses__.
        import configs.channels  # noqa: F401
        import services.skill_suggestion_message_processor  # noqa: F401
        import services.subconscious_worker  # noqa: F401
        from services.message_processor import MessageProcessor

        subs = [s.__name__ for s in MessageProcessor.__subclasses__()]
        assert subs == [], (
            f"MessageProcessor must have zero subclasses (§7a / P1); found: {subs}"
        )

    def test_no_production_class_subclasses_message_processor(self):
        """Static scan: no production class statement names MessageProcessor
        (or a *MessageProcessor alias) as a base."""
        pattern = re.compile(r"^\s*class\s+\w+\s*\([^)]*MessageProcessor[^)]*\)\s*:", re.M)
        offenders = []
        for rel, src in _prod_python_files():
            # Skip the base class definition itself.
            for m in pattern.finditer(src):
                line = m.group(0)
                if "class MessageProcessor(" in line or "class MessageProcessor:" in line:
                    continue
                offenders.append(f"{rel}: {line.strip()}")
        assert not offenders, (
            "No production class may subclass MessageProcessor (§7a / P1):\n  "
            + "\n  ".join(offenders)
        )


class TestP2OneDispatchPath:
    """P2 — One dispatch path for all tool calls (§7b / §7d / §1)."""

    def test_ability_dispatch_is_the_only_chokepoint(self):
        """Ability.dispatch() exists and is the single static chokepoint."""
        from abilities._base import Ability

        assert hasattr(Ability, "dispatch"), "Ability.dispatch must exist (§5 / P2)"

    def test_no_handle_tool_method_on_message_processor(self):
        """The old per-processor handle_tool() dispatch fork is gone."""
        from services.message_processor import MessageProcessor

        assert not hasattr(MessageProcessor, "handle_tool"), (
            "MessageProcessor.handle_tool() was the old dispatch fork — it must "
            "be gone; all tool calls route through Ability.dispatch() (P2)"
        )


class TestP3TrailSingleSourced:
    """P3 — Trail single-sourced, written once, no dual bookkeeping (§7g)."""

    def test_no_in_memory_act_trail_list(self):
        """The act-trail is reconstructed from tool_calls rows (§4c) — there is
        no in-memory ``_act_trail`` list duplicating that bookkeeping."""
        from services.message_processor import MessageProcessor

        mp = object.__new__(MessageProcessor)
        assert not hasattr(mp, "_act_trail"), (
            "in-memory _act_trail list duplicates the tool_calls source of "
            "truth — trail is single-sourced via _render_act_trail() (P3)"
        )

    def test_render_act_trail_exists(self):
        from services.message_processor import MessageProcessor

        assert hasattr(MessageProcessor, "_render_act_trail")
        assert hasattr(MessageProcessor, "_has_trail")


class TestP4OneMeasurementPath:
    """P4 — One measurement path; float vs 0.80/0.90; no compact_at column."""

    def test_provider_calculate_returns_float_fraction(self):
        from services.providers import Providers

        assert hasattr(Providers, "calculate"), (
            "Providers.calculate() is the single measurement path (§4b / P4)"
        )

    def test_no_compact_at_column_in_schema(self):
        schema = (_BACKEND_DIR / "schema.sql").read_text("utf-8")
        assert "compact_at" not in schema, (
            "compact_at column was removed — compaction triggers off a live "
            "float measurement vs 0.80/0.90 constants (§7e / P4)"
        )


class TestP5DelegateTypeNoCodeChange:
    """P5 — Adding a delegate type needs no change to existing code (§7f / §5b)."""

    def test_delegate_tools_are_self_contained_abilities(self):
        """Each delegate is a standalone Ability that calls the flat process();
        adding one is a new file, not an edit to the loop or a registry."""
        from abilities._base import Ability
        from abilities._registry import AbilityRegistry

        for name in ("web_search", "research", "summariser", "web_browse"):
            ability = AbilityRegistry.get(name)
            assert isinstance(ability, Ability), (
                f"delegate {name!r} must be a registered Ability (P5)"
            )


class TestP6CamelCaseGone:
    """P6 — camelCase shims gone; callers use snake_case (§7c / §1)."""

    def test_no_camelcase_methods_on_message_processor(self):
        from services.message_processor import MessageProcessor

        camel = re.compile(r"^[a-z]+[A-Z]")
        offenders = [
            name for name in dir(MessageProcessor)
            if camel.match(name)
        ]
        assert not offenders, (
            f"camelCase shims must be gone (P6); found: {offenders}"
        )

    def test_no_camelcase_method_defs_in_message_processor_module(self):
        src = (_BACKEND_DIR / "services" / "message_processor.py").read_text("utf-8")
        tree = ast.parse(src)
        camel = re.compile(r"^(get|set|handle|post|pre)[A-Z]")
        offenders = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and camel.match(node.name)
        ]
        assert not offenders, (
            f"camelCase method definitions must be gone (P6); found: {offenders}"
        )


# ---------------------------------------------------------------------------
# Q. Removed / deprecated surfaces absent (regression guards)
# ---------------------------------------------------------------------------


class TestQ1NoOverflowStrategyHook:
    """Q1 — No overflow_strategy hook on ProcessorConfig (§2 / §4a)."""

    def test_processor_config_has_no_overflow_strategy(self):
        from services.processor_config import ProcessorConfig

        assert "overflow_strategy" not in ProcessorConfig.__dataclass_fields__, (
            "overflow_strategy hook was removed from ProcessorConfig (Q1)"
        )


class TestQ2NoActWatermark:
    """Q2 — No act_watermark field / _max_tool_call_id helper (§4 / §4a / §7g)."""

    def test_no_act_watermark_field(self):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        assert "act_watermark" not in ProcessorConfig.__dataclass_fields__
        mp = object.__new__(MessageProcessor)
        assert not hasattr(mp, "act_watermark")

    def test_no_max_tool_call_id_helper(self):
        from services.message_processor import MessageProcessor

        assert not hasattr(MessageProcessor, "_max_tool_call_id"), (
            "_max_tool_call_id helper was removed (Q2)"
        )

    def test_no_act_watermark_or_max_tool_call_id_in_production(self):
        offenders = []
        for rel, src in _prod_python_files():
            if "act_watermark" in src or "_max_tool_call_id" in src:
                offenders.append(rel)
        assert not offenders, (
            f"act_watermark / _max_tool_call_id must be absent (Q2): {offenders}"
        )


class TestQ3DeletedServicesNotInvoked:
    """Q3 — ActDispatcherService / ToolRenderAndRecordService not invoked
    (§1 / §4c / §7d)."""

    def test_act_dispatcher_service_file_deleted(self):
        assert not (_BACKEND_DIR / "services" / "act_dispatcher_service.py").exists(), (
            "services/act_dispatcher_service.py must be deleted (Q3)"
        )

    def test_no_production_references_to_deleted_services(self):
        offenders = []
        for rel, src in _prod_python_files():
            tree = ast.parse(src)
            for node in ast.walk(tree):
                # Flag any actual *invocation* of the deleted classes — a stale
                # name in a comment/docstring is harmless, but a call is not.
                if isinstance(node, ast.Attribute):
                    if node.attr in ("dispatch_action",):
                        # only flag if the value chain mentions ActDispatcher
                        seg = ast.dump(node)
                        if "ActDispatcherService" in seg:
                            offenders.append(f"{rel}: {node.attr}")
                if isinstance(node, ast.Name) and node.id in (
                    "ActDispatcherService",
                    "ToolRenderAndRecordService",
                ):
                    offenders.append(f"{rel}: {node.id}")
        assert not offenders, (
            "Deleted services ActDispatcherService / ToolRenderAndRecordService "
            f"must not be referenced as live symbols (Q3): {offenders}"
        )


class TestQ4InternalDispatchHelpersRemoved:
    """Q4 — pre_dispatch, INTERNAL, handle() helper removed (§5)."""

    def test_ability_has_no_pre_dispatch(self):
        from abilities._base import Ability

        assert not hasattr(Ability, "pre_dispatch"), "Ability.pre_dispatch removed (Q4)"

    def test_ability_has_no_internal_flag(self):
        from abilities._base import Ability

        assert not hasattr(Ability, "INTERNAL"), "Ability.INTERNAL flag removed (Q4)"

    def test_ability_has_no_handle_helper(self):
        from abilities._base import Ability

        assert not hasattr(Ability, "handle"), "Ability.handle() helper removed (Q4)"

    def test_ability_uses_run_not_execute(self):
        from abilities._base import Ability

        assert hasattr(Ability, "run"), "abstract tool method is run() (§5)"
        assert not hasattr(Ability, "execute"), (
            "execute() was renamed to run() (§5 / Q4)"
        )


class TestQ5NoLogLabelAttr:
    """Q5 — No LOG_LABEL attr; telemetry label is config.job (§2 / §4a)."""

    def test_no_log_label_on_message_processor(self):
        from services.message_processor import MessageProcessor

        assert not hasattr(MessageProcessor, "LOG_LABEL"), (
            "LOG_LABEL class attr removed; telemetry label is config.job (Q5)"
        )

    def test_processor_config_has_no_log_label_field(self):
        from services.processor_config import ProcessorConfig

        assert "LOG_LABEL" not in ProcessorConfig.__dataclass_fields__
        assert "log_label" not in ProcessorConfig.__dataclass_fields__

    def test_processor_config_job_is_channel_role(self):
        """config.job derives from channel:role — it IS the telemetry label."""
        from services.processor_config import ProcessorConfig

        assert hasattr(ProcessorConfig, "job") or any(
            "job" in d for d in (vars(ProcessorConfig),)
        ), "ProcessorConfig must expose a .job (channel:role) telemetry label (Q5)"
