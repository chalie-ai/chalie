"""routes — the single place a URL path is decided.

Every ``/api/*`` route the :class:`~api.endpoint.Endpoint` contract generates
comes from the table below. A controller never names its own path: ``Endpoint``
takes its slug and :class:`~api.action.Action` its slug and verb as constructor
arguments, so the same controller can mount under more than one path and a path
change never touches controller code.

Each entry generates the routes its base class documents:

- ``Endpoint(slug)``       → ``/api/{slug}/all``, ``/api/{slug}/<id>``
- ``Action(slug, verb)``   → ``/api/{slug}/{verb}``, ``/api/{slug}/{verb}/<id>``

Controllers hold no instance state, so one instance each is built here at import
and shared by every app :func:`~api.create_app` produces. Adding a class under
``api/endpoints/`` or ``api/actions/`` does nothing until it is listed here —
``tests/test_routes.py`` fails on any controller this table forgets, so an
unmounted controller is a test failure rather than a silently missing URL.
"""

from __future__ import annotations

from .actions.capabilities.disconnect import CapabilitiesDisconnect
from .actions.capabilities.setup import CapabilitiesSetup
from .actions.capabilities.status import CapabilitiesStatus
from .actions.files.preview import FilesPreview
from .actions.lists.items import ListItems
from .actions.mcp_clients.discoverable import McpDiscoverable
from .actions.mcp_clients.test import McpTest
from .actions.mcp_clients.tools import McpTools
from .actions.mcp_server.regenerate_token import RegenerateTokenAction
from .actions.memory.search import MemorySearch
from .actions.policies.blocked import BlockedAction
from .actions.policies.reset import ResetAction
from .actions.policies.respond import RespondAction
from .actions.providers.catalog import ProviderCatalog
from .actions.providers.delegate import ProviderDelegate
from .actions.providers.list_models import ProviderListModels
from .actions.providers.selected import ProviderSelected
from .actions.providers.test import ProviderTest
from .actions.providers.vision import ProviderVision
from .actions.scheduler.turns import SchedulerTurns
from .actions.settings.personality import PersonalityAction
from .actions.skills.associations import SkillAssociations
from .actions.skills.copy import SkillCopy
from .actions.skills.toggle import SkillToggle
from .actions.snapshot.export import SnapshotExportAction
from .actions.snapshot.import_ import SnapshotImportAction
from .actions.voice.health import VoiceHealthAction
from .actions.voice.transcribe import VoiceTranscribeAction
from .actions.voice.transcript import VoiceTranscriptAction
from .endpoint import Endpoint
from .endpoints.capabilities import CapabilitiesEndpoint
from .endpoints.lists import Lists
from .endpoints.mcp_clients import McpClients
from .endpoints.mcp_settings import McpSettingsEndpoint
from .endpoints.policies import PoliciesEndpoint
from .endpoints.providers import Providers
from .endpoints.scheduler import Scheduler
from .endpoints.skills import Skills
from .endpoints.subagents import SubagentsEndpoint
from .endpoints.wrappers import WrappersEndpoint

ROUTES: tuple[Endpoint, ...] = (
    CapabilitiesEndpoint("capabilities"),
    CapabilitiesDisconnect("capabilities", "disconnect"),
    CapabilitiesSetup("capabilities", "setup"),
    CapabilitiesStatus("capabilities", "status"),

    FilesPreview("files", "preview"),

    Lists("lists"),
    ListItems("lists", "items"),

    McpClients("mcp-clients"),
    McpDiscoverable("mcp-clients", "discoverable"),
    McpTest("mcp-clients", "test"),
    McpTools("mcp-clients", "tools"),

    McpSettingsEndpoint("mcp-server"),
    RegenerateTokenAction("mcp-server", "regenerate-token"),

    MemorySearch("memory", "search"),

    PoliciesEndpoint("policies"),
    BlockedAction("policies", "blocked"),
    ResetAction("policies", "reset"),
    RespondAction("policies", "respond"),

    Providers("providers"),
    ProviderCatalog("providers", "catalog"),
    ProviderDelegate("providers", "delegate"),
    ProviderListModels("providers", "list-models"),
    ProviderSelected("providers", "selected"),
    ProviderTest("providers", "test"),
    ProviderVision("providers", "vision"),

    Scheduler("scheduler"),
    SchedulerTurns("scheduler", "turns"),

    PersonalityAction("settings", "personality"),

    Skills("skills"),
    SkillAssociations("skills", "associations"),
    SkillCopy("skills", "copy"),
    SkillToggle("skills", "toggle"),

    SnapshotExportAction("snapshot", "export"),
    SnapshotImportAction("snapshot", "import"),

    SubagentsEndpoint("subagents"),

    VoiceHealthAction("voice", "health"),
    VoiceTranscribeAction("voice", "transcribe"),
    VoiceTranscriptAction("voice", "transcript"),

    WrappersEndpoint("wrappers"),
)
"""Every mounted controller, grouped by slug. Order is presentational only —
Flask matches on the generated rule, never on registration order."""
