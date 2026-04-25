"""
Regression tests — webhook + digest-worker pipeline removal (2026-04-12).

Every test asserts that a specific piece of the legacy pipeline is gone and
cannot silently come back.  Tests inspect module importability, source text,
and live HTTP endpoints.  No behaviour is tested via mocks — we test real
absence.

Deleted surface:
  - workers/digest_worker.py
  - workers/digest_singletons.py
  - services/prompt_queue.py
  - services/system_prompt_assembly_service.py
  - services/user_prompt_assembly_service.py
  - services/context_relevance_service.py
  - services/orchestrator_service.py (+ services/orchestrator/ sub-package)
  - services/frontal_cortex_service.py
  - services/prompt_assembly_contract.py
  - services/goal_proactive_service.py
  - services/response_generation_service.py
  - services/prompt_assembly_service.py
  - POST /tools/webhook/<name>  (api/tools.py)
  - POST /tools/<name>/webhook/key  (api/tools.py)
  - ToolRegistryService.invoke_webhook()
  - ToolConfigService.generate/validate webhook methods + _webhook_key RESERVED_KEY
  - post_exchange_hooks._run_iip_hook / _run_belief_correction_hook / _classify_engagement
  - services/goal_proactive_service.py  (GoalProactiveService entirely)

All tests are marked ``unit`` — no network, no external processes.
"""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


# ── Part A: Module importability — all 12 deleted modules must raise ───────────

class TestDeletedModulesNotImportable:

    def _assert_gone(self, module_path: str):
        """Import the named module; assert it raises ModuleNotFoundError."""
        # Evict any cached stub so we test the real filesystem state.
        sys.modules.pop(module_path, None)
        with pytest.raises(ModuleNotFoundError):
            __import__(module_path)

    def test_digest_worker_cannot_be_imported(self):
        """workers.digest_worker must not exist — it was the legacy Phase A-E pipeline."""
        self._assert_gone("workers.digest_worker")

    def test_digest_singletons_cannot_be_imported(self):
        """workers.digest_singletons must not exist — singleton accessors for dead services."""
        self._assert_gone("workers.digest_singletons")

    def test_prompt_queue_cannot_be_imported(self):
        """services.prompt_queue must not exist — PromptQueue thread dispatcher deleted."""
        self._assert_gone("services.prompt_queue")

    def test_system_prompt_assembly_service_cannot_be_imported(self):
        """services.system_prompt_assembly_service must not exist — only caller was digest_worker."""
        self._assert_gone("services.system_prompt_assembly_service")

    def test_user_prompt_assembly_service_cannot_be_imported(self):
        """services.user_prompt_assembly_service must not exist — only caller was digest_worker."""
        self._assert_gone("services.user_prompt_assembly_service")

    def test_context_relevance_service_cannot_be_imported(self):
        """services.context_relevance_service must not exist — 7-layer gate, no surviving callers."""
        self._assert_gone("services.context_relevance_service")

    def test_orchestrator_service_cannot_be_imported(self):
        """services.orchestrator_service must not exist — path-based router, only used by digest_singletons."""
        self._assert_gone("services.orchestrator_service")

    def test_frontal_cortex_service_cannot_be_imported(self):
        """services.frontal_cortex_service must not exist — facade deleted with digest_worker."""
        self._assert_gone("services.frontal_cortex_service")

    def test_prompt_assembly_contract_cannot_be_imported(self):
        """services.prompt_assembly_contract must not exist — ABC with no surviving implementors."""
        self._assert_gone("services.prompt_assembly_contract")

    def test_goal_proactive_service_cannot_be_imported(self):
        """services.goal_proactive_service must not exist — GoalProactiveService deleted entirely."""
        self._assert_gone("services.goal_proactive_service")

    def test_response_generation_service_cannot_be_imported(self):
        """services.response_generation_service must not exist — delegated from FrontalCortexService."""
        self._assert_gone("services.response_generation_service")

    def test_prompt_assembly_service_cannot_be_imported(self):
        """services.prompt_assembly_service must not exist — only caller was FrontalCortexService."""
        self._assert_gone("services.prompt_assembly_service")


# ── Part B: __init__ exports are clean ─────────────────────────────────────────

class TestInitExportsClean:

    def test_services_init_has_no_PromptQueue(self):
        """services.__init__ must not export PromptQueue."""
        import services as svc_pkg
        assert not hasattr(svc_pkg, "PromptQueue"), (
            "PromptQueue still exported from services.__init__ — "
            "remove the 'from .prompt_queue import PromptQueue' line"
        )

    def test_services_init_has_no_FrontalCortexService(self):
        """services.__init__ must not export FrontalCortexService."""
        import services as svc_pkg
        assert not hasattr(svc_pkg, "FrontalCortexService"), (
            "FrontalCortexService still exported from services.__init__ — "
            "remove the 'from .frontal_cortex_service import ...' line"
        )

    def test_services_init_has_no_OrchestratorService(self):
        """services.__init__ must not export OrchestratorService."""
        import services as svc_pkg
        assert not hasattr(svc_pkg, "OrchestratorService"), (
            "OrchestratorService still exported from services.__init__ — "
            "remove the 'from .orchestrator_service import ...' line"
        )

    def test_workers_init_has_no_digest_worker(self):
        """workers.__init__ must not export digest_worker."""
        import workers as wrk_pkg
        assert not hasattr(wrk_pkg, "digest_worker"), (
            "digest_worker still exported from workers.__init__ — "
            "remove the 'from .digest_worker import digest_worker' line"
        )


# ── Part C: Orchestrator sub-package .py files deleted ─────────────────────────

class TestOrchestratorSubPackageDeleted:

    def test_orchestrator_handlers_py_does_not_exist(self):
        """services/orchestrator/handlers.py must not exist."""
        assert not (_BACKEND_ROOT / "services" / "orchestrator" / "handlers.py").exists(), (
            "services/orchestrator/handlers.py still exists — "
            "the orchestrator sub-package was supposed to be fully deleted"
        )

    def test_orchestrator_path_schemas_py_does_not_exist(self):
        """services/orchestrator/path_schemas.py must not exist."""
        assert not (_BACKEND_ROOT / "services" / "orchestrator" / "path_schemas.py").exists(), (
            "services/orchestrator/path_schemas.py still exists — "
            "delete it along with the orchestrator sub-package"
        )

    def test_orchestrator_init_py_does_not_exist(self):
        """services/orchestrator/__init__.py must not exist."""
        assert not (_BACKEND_ROOT / "services" / "orchestrator" / "__init__.py").exists(), (
            "services/orchestrator/__init__.py still exists — "
            "delete it along with the orchestrator sub-package"
        )


# ── Part D: Context-relevance config file deleted ──────────────────────────────

class TestContextRelevanceConfigDeleted:

    def test_context_relevance_json_does_not_exist(self):
        """backend/configs/agents/context-relevance.json must not exist."""
        path = _BACKEND_ROOT / "configs" / "agents" / "context-relevance.json"
        assert not path.exists(), (
            "configs/agents/context-relevance.json still exists — "
            "it was only loaded by the deleted ContextRelevanceService"
        )


# ── Part E: ToolRegistryService.invoke_webhook() removed ───────────────────────

class TestToolRegistryServiceWebhookRemoval:

    def test_invoke_webhook_method_removed(self):
        """ToolRegistryService must not have invoke_webhook() — no webhook endpoints survive."""
        from services.tool_registry_service import ToolRegistryService
        assert not hasattr(ToolRegistryService, "invoke_webhook"), (
            "ToolRegistryService.invoke_webhook() still present — "
            "it must be deleted along with the webhook endpoint"
        )


# ── Part F: ToolConfigService webhook methods removed ──────────────────────────

class TestToolConfigServiceWebhookRemoval:

    def test_generate_webhook_key_removed(self):
        """ToolConfigService must not have generate_webhook_key()."""
        from services.tool_config_service import ToolConfigService
        assert not hasattr(ToolConfigService, "generate_webhook_key"), (
            "ToolConfigService.generate_webhook_key() still present — delete it"
        )

    def test_get_webhook_key_removed(self):
        """ToolConfigService must not have get_webhook_key()."""
        from services.tool_config_service import ToolConfigService
        assert not hasattr(ToolConfigService, "get_webhook_key"), (
            "ToolConfigService.get_webhook_key() still present — delete it"
        )

    def test_validate_webhook_key_removed(self):
        """ToolConfigService must not have validate_webhook_key()."""
        from services.tool_config_service import ToolConfigService
        assert not hasattr(ToolConfigService, "validate_webhook_key"), (
            "ToolConfigService.validate_webhook_key() still present — delete it"
        )

    def test_validate_webhook_hmac_removed(self):
        """ToolConfigService must not have validate_webhook_hmac()."""
        from services.tool_config_service import ToolConfigService
        assert not hasattr(ToolConfigService, "validate_webhook_hmac"), (
            "ToolConfigService.validate_webhook_hmac() still present — delete it"
        )

    def test_webhook_key_not_in_reserved_keys(self):
        """_webhook_key must not appear in ToolConfigService.RESERVED_KEYS."""
        from services.tool_config_service import ToolConfigService
        assert "_webhook_key" not in ToolConfigService.RESERVED_KEYS, (
            "'_webhook_key' still in ToolConfigService.RESERVED_KEYS — "
            "remove it; the webhook-key concept no longer exists"
        )


# ── Part G: post_exchange_hooks — dead hooks deleted ───────────────────────────

class TestPostExchangeHooksDeadFunctionsRemoved:

    def test_run_iip_hook_removed(self):
        """_run_iip_hook must not exist in post_exchange_hooks — wired only via digest_worker."""
        import workers.post_exchange_hooks as mod
        assert not hasattr(mod, "_run_iip_hook"), (
            "_run_iip_hook still in post_exchange_hooks — "
            "it was only called by the deleted digest_worker Phase A pipeline"
        )

    def test_run_belief_correction_hook_removed(self):
        """_run_belief_correction_hook must not exist in post_exchange_hooks."""
        import workers.post_exchange_hooks as mod
        assert not hasattr(mod, "_run_belief_correction_hook"), (
            "_run_belief_correction_hook still in post_exchange_hooks — "
            "it was only called by the deleted digest_worker pipeline"
        )

    def test_classify_engagement_removed(self):
        """_classify_engagement must not exist in post_exchange_hooks."""
        import workers.post_exchange_hooks as mod
        assert not hasattr(mod, "_classify_engagement"), (
            "_classify_engagement still in post_exchange_hooks — "
            "it was only called by the deleted digest_worker pipeline"
        )


# ── Part H: Webhook endpoints return 404 ──────────────────────────────────────

class TestWebhookEndpointsReturn404:
    """The two deleted webhook routes must return 404 (or 405), not 200/400."""

    @pytest.fixture
    def client(self):
        from flask import Flask
        from api.tools import tools_bp
        from unittest.mock import patch
        app = Flask(__name__)
        app.register_blueprint(tools_bp)
        app.config["TESTING"] = True
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield app.test_client()

    def test_post_tools_webhook_name_returns_404(self, client):
        """POST /tools/webhook/<name> must return 404 — the endpoint was deleted."""
        resp = client.post("/tools/webhook/some_tool")
        assert resp.status_code in (404, 405), (
            f"POST /tools/webhook/some_tool returned {resp.status_code}, expected 404/405 — "
            "the webhook receiver endpoint must be gone"
        )

    def test_post_tools_name_webhook_key_returns_404(self, client):
        """POST /tools/<name>/webhook/key must return 404 — generate_webhook_key deleted."""
        resp = client.post("/tools/some_tool/webhook/key")
        assert resp.status_code in (404, 405), (
            f"POST /tools/some_tool/webhook/key returned {resp.status_code}, expected 404/405 — "
            "the webhook-key generation endpoint must be gone"
        )


# ── Part I: GET /tools response has no webhook fields ─────────────────────────

class TestToolsListHasNoWebhookFields:
    """list_tools() must not inject webhook_url or webhook_key_set into any tool entry."""

    @pytest.fixture
    def client(self, db):
        from flask import Flask
        from api.tools import tools_bp
        from unittest.mock import patch, MagicMock
        app = Flask(__name__)
        app.register_blueprint(tools_bp)
        app.config["TESTING"] = True

        # Minimal registry with one on_demand tool — avoids real tool loading.
        fake_manifest = {
            "description": "test tool",
            "trigger": {"type": "on_demand"},
            "config_schema": {},
            "icon": "T",
        }
        fake_registry = MagicMock()
        fake_registry.tools = {"test_tool": {"manifest": fake_manifest}}

        with patch("services.auth_session_service.validate_session", return_value=True), \
             patch("services.tool_registry_service.ToolRegistryService",
                   return_value=fake_registry), \
             patch("services.database_service.get_shared_db_service", return_value=db):
            yield app.test_client()

    def test_tool_entry_has_no_webhook_url(self, client):
        """Tool entries in GET /tools must not contain a webhook_url field."""
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.get_json()
        for tool in data.get("tools", []):
            assert "webhook_url" not in tool, (
                f"Tool '{tool.get('name')}' still has webhook_url in GET /tools — "
                "remove the webhook enrichment block from list_tools()"
            )

    def test_tool_entry_has_no_webhook_key_set(self, client):
        """Tool entries in GET /tools must not contain a webhook_key_set field."""
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.get_json()
        for tool in data.get("tools", []):
            assert "webhook_key_set" not in tool, (
                f"Tool '{tool.get('name')}' still has webhook_key_set in GET /tools — "
                "remove the webhook enrichment block from list_tools()"
            )


# ── Part J: /ready has no PromptQueue check ────────────────────────────────────

class TestReadyEndpointNoPromptQueueCheck:
    """/ready source must not import or reference PromptQueue."""

    def test_system_py_source_has_no_prompt_queue_import(self):
        """api/system.py must contain no PromptQueue import."""
        source = (_BACKEND_ROOT / "api" / "system.py").read_text()
        assert "PromptQueue" not in source, (
            "'PromptQueue' still referenced in api/system.py — "
            "the /ready worker-import check must be removed"
        )
        assert "prompt_queue" not in source, (
            "'prompt_queue' still referenced in api/system.py — "
            "the /ready worker-import check must be removed"
        )

    def test_ready_response_has_no_workers_key(self, db):
        """GET /ready must not include a 'workers' key — worker check was removed."""
        from flask import Flask
        from api.system import system_bp
        from unittest.mock import patch, MagicMock
        from services.memory_store import MemoryStore

        mock_onnx_svc = MagicMock()
        mock_onnx_svc.ready = True

        mock_session = MagicMock()

        app = Flask(__name__)
        app.register_blueprint(system_bp)
        app.config["TESTING"] = True
        client = app.test_client()

        real_store = MemoryStore()
        with patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=real_store), \
             patch("services.onnx_inference_service.get_onnx_inference_service",
                   return_value=mock_onnx_svc), \
             patch("services.embedding_service._session", mock_session):
            resp = client.get("/ready")

        data = resp.get_json()
        assert "workers" not in data, (
            "'workers' key present in /ready response — "
            "the legacy PromptQueue worker check was supposed to be removed entirely"
        )


# ── Part K: /system/status has no prompt-queue in queues ──────────────────────

class TestSystemStatusNoPromptQueue:
    """/system/status must not report a prompt-queue depth."""

    def test_system_status_has_no_prompt_queue_key(self, db):
        from flask import Flask
        from api.system import system_bp
        from unittest.mock import patch
        from services.memory_store import MemoryStore

        app = Flask(__name__)
        app.register_blueprint(system_bp)
        app.config["TESTING"] = True
        client = app.test_client()

        real_store = MemoryStore()
        with patch("services.auth_session_service.validate_session", return_value=True), \
             patch("services.memory_client.MemoryClientService.create_connection",
                   return_value=real_store), \
             patch("services.database_service.get_shared_db_service", return_value=db):
            resp = client.get("/system/status")

        assert resp.status_code == 200
        data = resp.get_json()
        queues = data.get("queues", {})
        assert "prompt-queue" not in queues, (
            "'prompt-queue' still in /system/status queues — "
            "remove it; the queue no longer exists"
        )


# ── Part L: GoalProactiveService production callers are gone ──────────────────

class TestGoalProactiveServiceCallersGone:

    def test_no_production_file_imports_goal_proactive_service(self):
        """No production .py file under backend/ may import goal_proactive_service.

        Walks services/, workers/, api/, and the top-level backend files.
        """
        search_dirs = [
            _BACKEND_ROOT / "services",
            _BACKEND_ROOT / "workers",
            _BACKEND_ROOT / "api",
        ]
        for search_dir in search_dirs:
            for py_file in search_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                text = py_file.read_text()
                assert "goal_proactive_service" not in text, (
                    f"'goal_proactive_service' still imported in {py_file.relative_to(_BACKEND_ROOT)} — "
                    "GoalProactiveService was fully deleted"
                )


# ── Part M: prompt-queue push in production code is gone ──────────────────────

class TestPromptQueuePushGone:
    """No production code may push to 'prompt-queue' — the consumer (digest_worker) is gone."""

    def _production_py_files(self):
        for d in ("services", "workers", "api"):
            for py_file in (_BACKEND_ROOT / d).rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    yield py_file

    def test_no_production_file_pushes_to_prompt_queue(self):
        """rpush/lpush to 'prompt-queue' must not appear in any production file."""
        for py_file in self._production_py_files():
            text = py_file.read_text()
            assert "rpush('prompt-queue'" not in text and 'rpush("prompt-queue"' not in text, (
                f"rpush to 'prompt-queue' found in {py_file.relative_to(_BACKEND_ROOT)} — "
                "the consumer (digest_worker) is deleted; pushes are silently discarded"
            )
