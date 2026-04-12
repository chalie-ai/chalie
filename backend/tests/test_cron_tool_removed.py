"""
Regression tests — cron-tool infrastructure removal (2026-04-10).

Every test in this file asserts that a specific piece of the legacy cron-tool
pipeline is gone and cannot silently come back.  The tests inspect imported
modules and source files directly; they do not test behaviour through a public
API because there *is* no public API — the whole thing was deleted.

All tests are marked ``unit`` — no network, no file I/O beyond the local
source tree, no external processes.

Architecture decision backing these tests:
  /Users/dylangrech/.claude/projects/-Volumes-llm-Chalie/memory/feedback_no_cron_in_tools.md
"""

import inspect
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Root of the backend package so we can build absolute paths reliably.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


# ── Part A: ToolRegistryService ──────────────────────────────────────────────

class TestToolRegistryServiceCronRemoval:

    def test_cron_tool_worker_class_removed(self):
        """_CronToolWorker must not exist as a module-level name."""
        import services.tool_registry_service as mod
        assert not hasattr(mod, "_CronToolWorker"), (
            "_CronToolWorker class still present in tool_registry_service — "
            "the cron-tool worker was supposed to be deleted"
        )

    def test_get_cron_tools_method_removed(self):
        """ToolRegistryService.get_cron_tools() must not exist."""
        from services.tool_registry_service import ToolRegistryService
        assert not hasattr(ToolRegistryService, "get_cron_tools"), (
            "ToolRegistryService.get_cron_tools() still present — "
            "should have been deleted with cron-tool support"
        )

    def test_create_cron_worker_method_removed(self):
        """ToolRegistryService.create_cron_worker() must not exist."""
        from services.tool_registry_service import ToolRegistryService
        assert not hasattr(ToolRegistryService, "create_cron_worker"), (
            "ToolRegistryService.create_cron_worker() still present — "
            "should have been deleted with cron-tool support"
        )

    def test_register_interface_tool_hardcodes_on_demand(self):
        """register_interface_tool() must hardcode trigger.type = 'on_demand'.

        This is a deliberate architectural constraint (see feedback_no_cron_in_tools.md).
        The hardcoding prevents interface tools from declaring any other trigger type.
        If this string disappears, someone is trying to parameterise it — stop them.
        """
        from services.tool_registry_service import ToolRegistryService
        source = inspect.getsource(ToolRegistryService.register_interface_tool)
        assert '"on_demand"' in source or "'on_demand'" in source, (
            "register_interface_tool() no longer hardcodes 'on_demand' trigger type. "
            "The hardcoding is deliberate — do not parameterise it."
        )


# ── Part A: InteractionLogService ────────────────────────────────────────────

class TestInteractionLogServiceCronRemoval:

    def test_activity_event_types_excludes_cron_tool_executed(self):
        """_ACTIVITY_EVENT_TYPES must not contain 'cron_tool_executed'."""
        from services.interaction_log_service import _ACTIVITY_EVENT_TYPES
        assert "cron_tool_executed" not in _ACTIVITY_EVENT_TYPES, (
            "'cron_tool_executed' found in _ACTIVITY_EVENT_TYPES — "
            "this event type was deleted with cron-tool support"
        )


# ── Part A: introspect_skill source scan ─────────────────────────────────────

class TestIntrospectSkillCronRemoval:

    def test_introspect_skill_source_has_no_cron_tool_executed(self):
        """introspect_skill.py must contain no reference to 'cron_tool_executed'.

        RELEVANT_TYPES inside _reasoning_recent_actions is function-local,
        so we inspect the source file directly rather than importing the tuple.
        """
        import services.innate_skills.introspect_skill as mod
        source = Path(mod.__file__).read_text()
        assert "cron_tool_executed" not in source, (
            "'cron_tool_executed' still appears in introspect_skill.py — "
            "this string should have been removed with cron-tool support"
        )


# ── Part A: dmn_service source scan ──────────────────────────────────────────

class TestDmnServiceCronRemoval:

    def test_dmn_service_source_has_no_cron_tool_channel(self):
        """dmn_service.py must contain no 'cron_tool:' channel prefix.

        The SQL clause that filtered on the cron_tool: source prefix was
        removed. This string appearing in source would mean someone is routing
        cron_tool events again.
        """
        source = (_BACKEND_ROOT / "services" / "dmn_service.py").read_text()
        assert "cron_tool:" not in source, (
            "'cron_tool:' channel prefix found in dmn_service.py — "
            "this was removed with cron-tool support"
        )


# ── Part A: output_service source scan ───────────────────────────────────────

class TestOutputServiceCronRemoval:

    def test_output_service_source_has_no_cron_tool_channel(self):
        """output_service.py must contain no 'cron_tool:' channel prefix."""
        source = (_BACKEND_ROOT / "services" / "output_service.py").read_text()
        assert "cron_tool:" not in source, (
            "'cron_tool:' channel prefix found in output_service.py — "
            "this was removed with cron-tool support"
        )


# ── Part A: decay_engine_service source scan ──────────────────────────────────

class TestDecayEngineServiceCronRemoval:

    def test_decay_engine_has_no_cron_tool_durability_branch(self):
        """decay_engine_service.py must contain no 'cron_tool' durability branch.

        The sf_durability == 'cron_tool' conditional is gone. These two strings
        cover both the condition string and any related constant.
        """
        source = (_BACKEND_ROOT / "services" / "decay_engine_service.py").read_text()
        assert "cron_tool_updated" not in source, (
            "'cron_tool_updated' found in decay_engine_service.py — "
            "the cron_tool durability branch was supposed to be removed"
        )
        assert "'cron_tool'" not in source, (
            "String literal 'cron_tool' found in decay_engine_service.py — "
            "the cron_tool durability branch was supposed to be removed"
        )


# ── Part A: cognitive_jobs.json ───────────────────────────────────────────────

class TestCognitiveJobsJsonCronRemoval:

    def _load_jobs(self) -> list:
        config_path = _BACKEND_ROOT / "configs" / "cognitive_jobs.json"
        return json.loads(config_path.read_text()).get("jobs", [])

    def test_cognitive_jobs_json_has_no_frontal_cortex_scheduled_tool(self):
        """cognitive_jobs.json must not contain the retired frontal-cortex-scheduled-tool job."""
        job_ids = {job["id"] for job in self._load_jobs()}
        assert "frontal-cortex-scheduled-tool" not in job_ids, (
            "'frontal-cortex-scheduled-tool' job still present in cognitive_jobs.json — "
            "this job was removed with cron-tool support"
        )

    def test_cognitive_jobs_json_has_no_frontal_cortex_act(self):
        """cognitive_jobs.json must not contain the retired frontal-cortex-act job."""
        job_ids = {job["id"] for job in self._load_jobs()}
        assert "frontal-cortex-act" not in job_ids, (
            "'frontal-cortex-act' job still present in cognitive_jobs.json — "
            "this job was removed as part of the unified-path migration"
        )


# ── Part A: deleted prompt file ───────────────────────────────────────────────

class TestDeletedPromptFile:

    def test_frontal_cortex_scheduled_tool_prompt_deleted(self):
        """backend/prompts/frontal-cortex-scheduled-tool.md must not exist."""
        prompt_path = _BACKEND_ROOT / "prompts" / "frontal-cortex-scheduled-tool.md"
        assert not prompt_path.exists(), (
            "frontal-cortex-scheduled-tool.md still exists — "
            "it should have been deleted along with the cron-tool job config"
        )


# ── Part A: ScheduledMessageProcessor is the live alternative path ─────────────

class TestScheduledMessageProcessorExists:

    def test_scheduled_message_processor_is_message_processor_subclass(self):
        """ScheduledMessageProcessor must exist and subclass MessageProcessor.

        This guards the 'alternative path is still alive' invariant. If this
        import fails, the replacement for the deleted cron pipeline is gone too.
        """
        from services.scheduled_message_processor import ScheduledMessageProcessor
        from services.message_processor import MessageProcessor
        assert issubclass(ScheduledMessageProcessor, MessageProcessor), (
            "ScheduledMessageProcessor is not a subclass of MessageProcessor — "
            "the replacement scheduled-task path is broken"
        )
