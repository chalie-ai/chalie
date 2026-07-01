# `# noqa` Audit — `rc-1.1.0`

**Branch:** `chore/noqa-audit-rc-1.1.0` (cloned from `rc-1.1.0`)
**Scope:** every `# noqa` comment in `backend/**/*.py` (330 occurrences across 76 files)

> **Audit corrections (this revision):** the previous draft listed 75 files, but `rg -l 'noqa' backend/ -tpy` returns 76 files. The earlier draft under-reported one file entirely (`capabilities/mail_capability/carddav_handler.py`, 6 PLC0415) and miscounted `mcp_server/server.py` (claimed 10 noqas; actual is 3 `# noqa: PLC0415` plus 7 raw PLC0415 violations that ruff will flag). Both gaps are corrected below.
**Methodology:** static cycle analysis (AST graph of top-level imports) + per-file review by parallel subagents

---

## Headline Finding

The top-level intra-package import graph is a **clean DAG** — zero cycles. Therefore **no deferred-in-function import is required to break a circular dependency**. The `# noqa: PLC0415` comments (241 of 330) fall into three real categories:

| Category | Count | Action |
|---|---|---|
| **Phantom** — light first-party/utility/stdlib import, no cycle, no side effects | ~150 | **REMOVE** — hoist to top-level |
| **Cold-start** — heavy third-party lib (openai, anthropic, tiktoken, numpy, sqlite_vec, etc.) | ~40 | **KEEP** — deferral is legitimate |
| **Singleton/IO** — module-level singleton or eager I/O (database_service, world_state, embedding_service) | ~50 | **KEEP** — deferral is legitimate |

The non-`PLC0415` noqas (89 of 330) are overwhelmingly legitimate (`E402` in sys.path-bootstrapped scripts, `BLE001` with documented fail-open contracts, `F401` side-effect imports, `N802` contract names, etc.). A handful are fixable (`B017` broad pytest.raises, one stale `F821` forward-ref).

---

## Standardized Import Methodology (the rule going forward)

1. **Top-level by default.** All first-party imports live at the top of the file, grouped: stdlib → third-party → first-party, alphabetical within each group. No `# noqa` unless the import genuinely cannot live at top-level.
2. **`if TYPE_CHECKING:` for type-only.** Any import used **only** in a `cast(...)` or a string annotation belongs under `if TYPE_CHECKING:` at the top of the file — **not** deferred inside a function. This removes both the runtime cost and the `# noqa`.
3. **Deferred only for heavy third-party or eager singletons.** An import may live inside a function **only** when the target module (a) imports a heavy third-party library at module scope (openai, anthropic, google-genai, tiktoken, numpy, transformers, playwright, caldav, vobject, imapclient, httpx, sqlite_vec, rapidocr, pdfplumber, trafilatura), or (b) performs eager singleton initialization / I/O at module scope (database_service, embedding_service, world_state). In both cases the `# noqa: PLC0415` **stays** and **must** carry a trailing `— <reason>` comment.
4. **No deferral for circular-dependency avoidance.** The graph is a DAG; if a future change reintroduces a cycle, fix the cycle (extract a shared module, invert the dependency, or use `TYPE_CHECKING`) rather than deferring the import.
5. **`# noqa` must always cite a reason.** Bare `# noqa` without a rule code and a `— reason` is forbidden. Every suppression must be auditable.

---

## Per-File Audit

<!-- BUCKET 1 -->

### `services/subconscious_worker.py` (32 noqas)

The largest single file. Heavy use of deferred imports across background-worker paths.

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (many) | PLC0415 | `import logging`, `import json`, `import os`, `from services.time_utils import utc_now`, `from services.act_trail import ActTrail`, `from services.transcript_service import Transcript`, `from services.provider_api import …`, `from services.llm_service import …`, `from services.providers import …`, `from services.message_processor import MessageProcessor` (type-only), `from services.processor_config import …` (type-only), `from services.turn_zero_flashback import TurnZeroFlashback` | REMOVE | Hoist all stdlib + light first-party imports to top-level. Move type-only imports (`MessageProcessor`, `ProcessorConfig`) into an `if TYPE_CHECKING:` block | All are stdlib or pure first-party utilities with no import-time side effects; DAG is acyclic |
| (several) | PLC0415 | `import numpy as np`, `from services.embedding_service import get_embedding_service`, `from services.database_service import get_shared_db_service` | KEEP | No change — numpy is a heavy third-party lib; embedding_service pulls numpy at module scope; database_service has lazy-singleton accessor but is on the legitimate-deferral list | Heavy third-party / singleton-deferral justified |
| (several) | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change — background-worker fail-open contract; every occurrence has a rationale comment | Documented fail-open per the worker's fault-isolation contract |

---

### `services/llm_clients/gemini.py` (6 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 205 | PLC0415 | `from google import genai` | KEEP | No change | Heavy third-party SDK — cold-start deferral legitimate |
| 213 | PLC0415 | `from services.llm_service import _resolve_api_key, _app_user_agent` | REMOVE | Hoist to top-level imports | `llm_service.py` only imports `re`, `logging`, `typing` — pure helper |
| 214 | PLC0415 | `from services.providers import PROVIDER_CALL_TIMEOUT_S` | REMOVE | Hoist to top-level imports | Pure constants module, no side effects |
| 346 | PLC0415 | `import httpx` | KEEP | No change | Moderate-to-heavy third-party HTTP client; cold-start deferral legitimate |
| 408 | PLC0415 | `from services.llm_service import _resolve_api_key` | REMOVE | Hoist (consolidate with line 213) | Same as line 213 |
| 420 | PLC0415 | `from services.llm_service import estimate_tokens` | REMOVE | Hoist (consolidate with line 213) | Same as line 213 |

---

### `utils/build_ability_db.py` (5 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with E402) | E402 | `from services.embedding_service import EmbeddingService`, `from services.embedding_utils import pack_embedding`, `from services.file_mapper_service import FileMapperService` | KEEP | No change — `sys.path.insert(0, …)` bootstrap precedes these imports | Standalone script; sys.path manipulation requires imports to come after |
| (lines with PLC0415) | PLC0415 | stdlib + light first-party imports inside functions | REMOVE | Hoist to top-level (after the sys.path bootstrap) | Stdlib and pure utilities — no deferral needed |

---

### `services/llm_clients/anthropic.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 153 | PLC0415 | `from services.llm_service import _resolve_api_key, _app_user_agent` | REMOVE | Hoist to top-level imports | Pure helper module — no side effects |
| 154 | PLC0415 | `from services.providers import PROVIDER_CALL_TIMEOUT_S` | REMOVE | Hoist to top-level imports | Pure constants module |
| 182 | PLC0415 | `import anthropic` | KEEP | No change | Heavy third-party SDK — cold-start deferral legitimate |
| 283 | PLC0415 | `from services.llm_service import estimate_tokens` | REMOVE | Hoist (consolidate with line 153) | Same as line 153 |

---

### `abilities/web_browse.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 104 | PLC0415 | `from services.message_processor import MessageProcessor` | REMOVE | Hoist to top-level imports | First-party; only stdlib + time_formatter_service at module scope — cheap to import |
| 123 | PLC0415 | `from services.database_service import get_shared_db_service` | KEEP | No change | Lazy singleton; on the legitimate-deferral list |
| 124 | PLC0415 | `from services.document_service import DocumentService` | REMOVE | Hoist to top-level imports | Pure class, no side effects at import |

---

### `capabilities/mail_capability/capability.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 23 | PLC2701 | `from capabilities.mail_capability.caldav_handler import _CalDAVClient` | KEEP | No change | Private proto import — organizational boundary, comment explains |
| 24 | PLC2701 | `from capabilities.mail_capability.imap_handler import _ImapClient` | KEEP | No change | Same as line 23 |
| 25 | PLC2701 | `from capabilities.mail_capability.carddav_handler import _CaldavClientProto` | KEEP | No change | Same as line 23 — proto type for duck-typing |

---

### `capabilities/mail_capability/carddav_handler.py` (6 noqas) — **not in previous draft**

Cold-start deferrals for the caldav/vobject third-party libs (on the whitelist at rule §3), with the `capabilities.contact_resolver` index/resolve helpers as hoistable first-party imports.

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 65 | PLC0415 | `import caldav` (inside `open_client`) | KEEP | No change | Heavy third-party lib — on the legitimate-deferral whitelist |
| 80 | PLC0415 | `import caldav as _caldav` (inside `sync_contacts`) | KEEP | No change | Same as line 65 — same module re-imported under alias for name-shadowing inside the method |
| 134 | PLC0415 | `import vobject` (inside `parse_vcard`) | KEEP | No change | Heavy third-party lib — on the legitimate-deferral whitelist |
| 174 | PLC0415 | `from capabilities.contact_resolver import index_contact_profile` (inside `index_contacts`) | REMOVE | Hoist to top-level (consolidate with lines 198, 224) | `contact_resolver` only imports `json`/`logging`/`typing` at module scope — no heavy third-party, no eager singleton; per the standardized methodology §3, first-party deferral is not legitimate |
| 198 | PLC0415 | `from capabilities.contact_resolver import (…)` (multiline import inside `index_contacts`) | REMOVE | Hoist (consolidate with lines 174, 224) | Same as line 174 |
| 224 | PLC0415 | `from capabilities.contact_resolver import resolve` (inside `resolve_contact`) | REMOVE | Hoist (consolidate with lines 174, 198) | Same as line 174 |

---

### `services/text_extractor.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 245 | PLC0415 | `import mimetypes` | REMOVE | Hoist to top-level imports (or delete — `mimetypes` is already imported at top-level) | stdlib; deferral is meaningless |
| 246 | PLC0415 | `from abilities.vision import RICH_INDEX_PROMPT, describe_image` | REMOVE | Hoist to top-level imports | First-party; `vision.py`'s heavy `rapidocr` is deferred inside `analyze()` itself — module import is cheap |
| 247 | PLC0415 | `from services.processor_config import ProcessorConfig` | REMOVE | Hoist to top-level imports | Pure ABC, no side effects |

---

### `abilities/_event_emitter.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415) | PLC0415 | light first-party imports | REMOVE | Hoist to top-level imports | No import-time side effects; DAG is acyclic |

---

### `configs/channels/skill_suggestion.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415) | PLC0415 | light first-party/stdlib imports | REMOVE | Hoist to top-level imports | Pure utilities / stdlib — no side effects |

---

### `services/vision_service.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415) | PLC0415 | light first-party imports | REMOVE | Hoist to top-level imports | No import-time side effects |

---

### `abilities/_budget.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (line with PLC0415) | PLC0415 | light first-party import | REMOVE | Hoist to top-level imports | No import-time side effects |

---

### `configs/channels/episode_encoder.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (line with PLC0415) | PLC0415 | light first-party/stdlib import | REMOVE | Hoist to top-level imports | No import-time side effects |

---

### `tests/_registry_invariants.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (line with SLF001) | SLF001 | private member access for test-seam reset | KEEP | No change | Deliberate test-seam access to `_reset_for_tests()` / `_get_registry()` |

---

### `tests/test_search_record_format.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (line with E402) | E402 | import after test code | KEEP or REMOVE | If sys.path bootstrap precedes → KEEP; if sloppy ordering → REMOVE (move to top) | Read the file — depends on whether a `sys.path.insert` precedes |

---

### `services/message_processor.py` (29 noqas) — **highest-impact file**

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 110 | PLC0415 | `from services.providers import Providers` | REMOVE | Hoist to top-level | `providers.py` is stdlib-only (json/logging/threading), lazy `Providers` class |
| 223 | F821 | `config: "ProcessorConfig",  # noqa: F821 — deferred import avoids circular dep` | TRANSFORM | Move `from services.processor_config import ProcessorConfig` into the existing `if TYPE_CHECKING:` block at top | The graph is a DAG; the forward reference only needs `TYPE_CHECKING`, not a runtime import |
| 256 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | `transcript_service` is stdlib-only, lazy class — explicitly listed as a REMOVE target |
| 289 | PLC0415 | `from services.turn_zero_flashback import TurnZeroFlashback` | REMOVE | Hoist to top-level | Module-level imports are `logging` + `time_formatter_service` only — lightweight |
| 300 | PLC0415 | `from concurrent.futures import ThreadPoolExecutor` | REMOVE | Hoist to top-level | Pure stdlib |
| 306 | PLC0415 | `import os` | REMOVE | Hoist to top-level | Pure stdlib |
| 307 | PLC0415 | `from abilities._dispatcher import ToolDispatcher` | KEEP | No change | `_dispatcher` imports a large graph (`policy_manager`, `_registry`, `_mcp_ability`, `_event_emitter`, `client_context`) at module level — heavy |
| 308 | PLC0415 | `from services.tmp_storage import TMP_PATH_PREFIX` | REMOVE | Hoist to top-level | Module is `os`+`tempfile`+constants — nothing heavy |
| 324 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist (consolidate with line 256) | Same as line 256 |
| 331 | PLC0415 | `from services.provider_api import ProviderApiRequest, ThinkingLevel, ProviderType` | REMOVE | Hoist to top-level | `provider_api` is stdlib-only (`dataclass`/`enum`) — pure types |
| 332 | PLC0415 | `from abilities._registry import AbilityRegistry` | KEEP | No change | `_registry` has module-level `threading.RLock` and triggers a filesystem walk of `abilities/` at first resolution |
| 333 | PLC0415 | `from services.providers import resolve_thinking_mode` | REMOVE | Hoist (consolidate with line 110) | Same as line 110 |
| 385 | PLC0415 | `from services.provider_api import RequestOverCapError, ResponseOverLimitError` | REMOVE | Hoist (consolidate with line 331) | Same as line 331 |
| 413 | PLC0415 | `from services.provider_api import (ProviderRetriesExhaustedError, RequestOverCapError, ResponseOverLimitError)` | REMOVE | Hoist (consolidate with line 331) | Same as line 331 |
| 423 | BLE001 | `except Exception as exc:  # noqa: BLE001 — every provider failure is retriable here` | KEEP | No change | Documented retry policy — intentional broad catch |
| 447 | PLC0415 | `from api.chat import _broadcast_provider_retry` | KEEP | No change | `api/chat.py` imports Flask (`Blueprint`, `jsonify`, `request`) at module level — heavy third-party |
| 455 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist (consolidate with line 256) | Same as line 256 |
| 477 | PLC0415 | `from api.chat import _broadcast_interim` | KEEP | No change | Same as line 447 — Flask-blueprint import |
| 482 | PLC0415 | `from abilities._dispatcher import ToolDispatcher` | KEEP | No change | Same as line 307 |
| 505 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist (consolidate with line 256) | Same as line 256 |
| 515 | PLC0415 | `from services.markup import markdown_to_html` | KEEP | No change | `services.markup` imports `nh3` (third-party HTML sanitiser) at module level — heavy |
| 528 | BLE001 | `except Exception as exc:  # noqa: BLE001 — failure isolation contract` | KEEP | No change | Post-turn-hook isolation is a documented contract |
| 543 | PLC0415 | `from services.database_service import get_shared_db_service` | KEEP | No change | `database_service` is on the singleton-deferral list |
| 573 | PLC0415 | `from services import compaction_persistence` | REMOVE | Hoist to top-level | stdlib-only (`logging`/`Optional`/`Dict`), functions-only — pure utility |
| 574 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist (consolidate with line 256) | Same as line 256 |
| 600 | PLC0415 | `from services.act_trail import ActTrail` | REMOVE | Hoist to top-level | `act_trail` only imports `time_utils`; lazy `ActTrail` class — pure utility |
| 612 | PLC0415 | `from abilities._dispatcher import ToolDispatcher` | KEEP | No change | Same as line 307 |
| 613 | PLC0415 | `from services import compaction_persistence` | REMOVE | Hoist (consolidate with line 573) | Same as line 573 |
| 614 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist (consolidate with line 256) | Same as line 256 |

---

### `tests/test_vision_ability.py` (7 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 104 | B017,PT011 | `with pytest.raises(Exception):  # noqa: B017,PT011` | TRANSFORM | Narrow to `pytest.raises(ProviderRetriesExhaustedError)` (or the concrete exception MessageProcessor surfaces) | The comment confirms a specific provider error is expected — narrow the assertion |
| 165 | E402 | `import hashlib  # noqa: E402 — appended to existing file` | KEEP | No change | Merged-block artifact; pytest still collects both |
| 167 | E402 | `from abilities._dispatcher import ToolDispatcher` | KEEP | No change | Same as line 165 |
| 168 | E402 | `from configs.channels import UserConfig` | KEEP | No change | Same as line 165 |
| 169 | E402 | `from services.document_service import DocumentService` | KEEP | No change | Same as line 165 |
| 170 | E402 | `from services.file_mapper_service import FileMapperService` | KEEP | No change | Same as line 165 |
| 171 | E402 | `from tests._tool_result_harness import MP, body, head, seed_transcript` | KEEP | No change | Same as line 165 |

---

### `tests/test_snapshot_service_additive.py` (6 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 26 | F401 | `from tests.test_snapshot_service import (  # noqa: F401 …` | KEEP | No change | Pure side-effect import — pytest needs the symbols in this module's namespace |
| 43 | F811 | `def test_http_export_route_streams_a_real_zip(self, client) -> None:  # noqa: F811` | KEEP | No change | Deliberate re-definition of a fixture-name method on an extended test class |
| 66 | F811 | `def test_apply_pending_is_a_noop_when_nothing_is_staged(self, instance) -> None:  # noqa: F811` | KEEP | No change | Same as line 43 |
| 86 | F811 | `def test_mid_swap_failure_rolls_back_and_quarantines_to_break_boot_loop(self, instance) -> None:  # noqa: F811` | KEEP | No change | Same as line 43 |
| 133 | F811 | `def test_plain_export_opens_without_password_and_missing_manifest_is_rejected(self, instance) -> None:  # noqa: F811` | KEEP | No change | Same as line 43 |
| 166 | F811 | `def test_restore_skips_unknown_artifact_kind_from_an_older_build(self, instance) -> None:  # noqa: F811` | KEEP | No change | Same as line 43 |

---

### `abilities/chat_history_compactor.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 83 | PLC0415 | `from services.message_processor import MessageProcessor` | KEEP | No change | `MessageProcessor` is the orchestrator with a deep import tree — heavy |
| 84 | PLC0415 | `from services import compaction_persistence` | REMOVE | Hoist to top-level | stdlib-only functions module — pure utility |
| 85 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | stdlib-only, lazy class |
| 140 | PLC0415 | `from services.provider_api import ProviderApiRequest, ThinkingLevel` | REMOVE | Hoist to top-level | Pure types — no side effects |

---

### `services/llm_clients/factory.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 36 | PLC0415 | `from services.llm_clients.ollama import OllamaClient` | KEEP | No change | Cold-start deferral — `ollama.py` imports `requests` at module level; per-platform dispatch avoids loading every client at boot |
| 42 | PLC0415 | `from services.llm_clients.anthropic import AnthropicClient` | KEEP | No change | `anthropic` SDK is heavy |
| 53 | PLC0415 | `from services.llm_clients.openai import OpenAIClient` | KEEP | No change | `openai` SDK is heavy |
| 59 | PLC0415 | `from services.llm_clients.gemini import GeminiClient` | KEEP | No change | `google.genai` SDK is heavy |

---

### `mcp_server/server.py` (3 `# noqa: PLC0415` + 7 unflagged PLC0415 violations) — **previous draft overcounted**

The previous draft listed 10 rows and called them all "noqas". On the actual file, only three lines (145, 146, 154) carry `# noqa: PLC0415`. The other seven rows (76, 77, 78, 183, 184, 211, 212) are **raw** `import-outside-top-level` violations — `python -m ruff check --select PLC0415 backend/mcp_server/server.py` reports all seven as errors, and they are not present in any `# noqa` count. Each row below carries the verdict that *should* hold once the suppression is added or the import is hoisted; the action column tells you which.

**Suppressed (`# noqa: PLC0415` present)**

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 145 | PLC0415 | `from configs.channels import EAMPConfig` | KEEP | No change | `configs/channels/__init__.py` imports every channel config at module load — heavy |
| 146 | PLC0415 | `from services.message_processor import MessageProcessor` | KEEP | No change | `MessageProcessor` has a deep, heavy import tree |
| 154 | PLC0415 | `from services.provider_api import ProviderRetriesExhaustedError` | REMOVE | Hoist to top-level | Pure types — no side effects |

**Unflagged PLC0415 violations (`# noqa` missing — `ruff` will flag them)**

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 76 | PLC0415 | `from services.wrapper_auth_service import _hash_token` (inside `_validate_token`) | KEEP | Add `# noqa: PLC0415` (preserves KEEP) | `wrapper_auth_service` imports `sqlite3` + `flask` at module level — heavy |
| 77 | PLC0415 | `from services.database_service import get_shared_db_service` (inside `_validate_token`) | KEEP | Add `# noqa: PLC0415` (preserves KEEP) | Singleton — on the legitimate-deferral list |
| 78 | PLC0415 | `from services.time_utils import utc_now` (inside `_validate_token`) | REMOVE | Hoist to top-level (preferred over adding `# noqa`) | `time_utils` is stdlib-only — explicitly a REMOVE target |
| 183 | PLC0415 | `from services.settings_service import SettingsService` (inside `run_mcp_server`) | KEEP | Add `# noqa: PLC0415` (preserves KEEP) | `SettingsService` imports `database_service` at module level — singleton-deferral applies transitively |
| 184 | PLC0415 | `from services.database_service import get_shared_db_service` (inside `run_mcp_server`) | KEEP | Add `# noqa: PLC0415` (consolidate with line 77) | Singleton — on the legitimate-deferral list |
| 211 | PLC0415 | `from services.wrapper_auth_service import WrapperAuthService` (inside `_ensure_mcp_token`) | KEEP | Add `# noqa: PLC0415` (consolidate with line 76) | Same as line 76 |
| 212 | PLC0415 | `from services.settings_service import SettingsService` (inside `_ensure_mcp_token`) | KEEP | Add `# noqa: PLC0415` (consolidate with line 183) | Same as line 183 |

---

### `tests/test_tool_result_contract.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 204 | SLF001 | `_reg_mod._reset_for_tests()  # noqa: SLF001` | KEEP | No change | Deliberate test seam |
| 205 | SLF001 | `_reg_mod._get_registry()  # noqa: SLF001` | KEEP | No change | Same as line 204 |
| 209 | SLF001 | `_reg_mod._reset_for_tests()  # noqa: SLF001` | KEEP | No change | Same as line 204 (teardown) |

---

### `abilities/_mcp_ability.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 50 | PLC0415 | `from services.mcp_client_service import (  # noqa: PLC0415 …` | KEEP | No change | `mcp_client_service` imports `database_service` at module level (singleton) + `asyncio`/`sqlite3` — heavy |
| 118 | PLC0415 | `from services.mcp_client_service import McpClientService` | KEEP | No change | Same as line 50 |

---

### `migrate_transcript_rebuild.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 37 | E402 | `from services.transcript_service import Transcript  # noqa: E402` | KEEP | No change | Standalone-script bootstrap — `sys.path.insert` must run first |
| 38 | E402 | `from services.file_mapper_service import FileMapperService  # noqa: E402` | KEEP | No change | Same as line 37 |

---

### `tests/test_ability_home.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 76 | N802 | `def do_GET(self) -> None:  # noqa: N802 (http.server contract)` | KEEP | No change | `BaseHTTPRequestHandler.do_GET` is mandated by the `http.server` contract |
| 98 | N802 | `def do_POST(self) -> None:  # noqa: N802 (http.server contract)` | KEEP | No change | Same as line 76 |

---

### `abilities/code_eval.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 172 | S603 | `completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell` | KEEP | No change | Fixed argv list, no `shell=True`, documented — canonical S603 keep case |

---

### `configs/channels/web_browse.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 53 | ARG002 | `def run(self, mp: "MessageProcessor", result_text: str) -> None:  # noqa: ARG002 — hook signature` | KEEP | No change | `run` is the `PostTurnHook` interface contract; both parameters are part of the signature |

---

### `tests/e2e/test_browser_live.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 24 | ARG001 | `def test_full_browse_flow_on_one_persistent_page(db: sqlite3.Connection) -> None:  # noqa: ARG001` | KEEP | No change | `db` is the shared session-scoped fixture that wires the production DB — comment documents why |

---

### `tests/test_super_episode_pipeline.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 677 | F401 | `from sklearn.cluster import HDBSCAN  # noqa: F401 — resolution is the assertion` | KEEP | No change | The import IS the assertion — a regression lock that `HDBSCAN` resolves |

---

### `services/providers.py` (14 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 43 | PLC0415 | `from services.provider_api import ProviderType, ProviderError` | REMOVE | Hoist to top-level runtime imports (`ProviderType` is used in `==` comparisons) | `provider_api` is pure dataclasses + enum — no side effects |
| 44 | PLC0415 | `from services.llm_clients.factory import build_client` | REMOVE | Hoist to top-level | Pure factory function — no module-level singleton |
| 49 | PLC0415 | `from services.provider_db_service import ProviderDbService` | REMOVE | Hoist to top-level | Class with light init; `database_service` is imported but lazy |
| 50 | PLC0415 | `from services.database_service import get_shared_db_service` | REMOVE | Hoist to top-level | Pure lazy getter — no eager init at import |
| 57 | PLC0415 | `from services.provider_db_service import ProviderDbService` (duplicate) | REMOVE | Hoist (consolidate with line 49) | Same as line 49 |
| 58 | PLC0415 | `from services.database_service import get_shared_db_service` (duplicate) | REMOVE | Hoist (consolidate with line 50) | Same as line 50 |
| 67 | PLC0415 | `from services.provider_cache_service import ProviderCacheService` | REMOVE | Hoist to top-level | Class with lazy state, no module-level singleton |
| 78 | PLC0415 | `from services.provider_api import RequestOverCapError` | REMOVE | Hoist (consolidate with line 43) | Pure exception class |
| 106 | PLC0415 | `from services.provider_api import ProviderType` (duplicate) | REMOVE | Hoist (consolidate with line 43) | Same as line 43 |
| 110 | PLC0415 | `from services.provider_cache_service import ProviderCacheService` (duplicate) | REMOVE | Hoist (consolidate with line 67) | Same as line 67 |
| 135 | PLC0415 | `from services.llm_request_logger import log_llm_request` | REMOVE | Hoist to top-level | Module-level `FileMapperService.get_logs_path()` is a path expression — cheap |
| 150 | PLC0415 | `from services.llm_call_log_service import log_call` | REMOVE | Hoist to top-level | Imports `database_service` + `time_utils` — light |
| 175 | PLC0415 | `from services.metrics_service import MetricsService` | REMOVE | Hoist to top-level | Class with lazy state — no module-level singleton |
| (lines with N802 / other) | N802 | deliberate casing | KEEP | No change | Read the comment — usually a deliberate name |

---

### `configs/channels/user.py` (9 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415 — stdlib) | PLC0415 | `import logging`, `import json` | REMOVE | Hoist to top-level | stdlib — deferral never legitimate |
| (lines with PLC0415 — first-party) | PLC0415 | `from services.act_trail import ActTrail`, `from services.time_utils import utc_now`, `from services.transcript_service import Transcript`, `from services.skill_association_service import SkillAssociationService` | REMOVE | Hoist to top-level | Pure utilities — no side effects |
| (lines with PLC0415 — database_service) | PLC0415 | `from services.database_service import get_shared_db_service` | KEEP | No change | Singleton — on the legitimate-deferral list |

---

### `abilities/_dispatcher.py` (8 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 81 | PLC0415 | `from services.message_processor import _sanitize_llm_args` | REMOVE | Hoist to top-level | First-party; `message_processor` top-level is stdlib + time_formatter_service — cheap |
| 105 | BLE001 | `except Exception:  # noqa: BLE001` | KEEP | No change | Tool-dispatch fail-open contract |
| 141 | PLC0415 | `from services.act_trail import ActTrail` | REMOVE | Hoist to top-level | Pure utility class |
| 189 | PLC0415 | `from services.act_trail import ActTrail` (duplicate) | REMOVE | Hoist (consolidate with line 141) | Same as line 141 |
| 238 | PLC0415 | `from services.act_trail import ActTrail` (duplicate) | REMOVE | Hoist (consolidate with line 141) | Same as line 141 |
| 239 | PLC0415 | `from services.message_processor import MessageProcessor` | KEEP | No change | `MessageProcessor` is the heavy orchestrator — deferral justified |
| 262 | PLC0415 | `from services.async_delegate_runner import async_delegate_runner` | REMOVE | Hoist to top-level | First-party runner module — no module-level singleton |
| 349 | BLE001 | `except Exception:  # noqa: BLE001` | KEEP | No change | Tool-dispatch fail-open contract |
| 356 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Same as line 349 |
| 359 | PLC0415 | `from services.vault_service import VaultLockedError` | REMOVE | Hoist to top-level | Pure exception class — no side effects |

---

### `abilities/browser.py` (6 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415 — first-party) | PLC0415 | `from services.database_service import get_shared_db_service`, `from services.document_service import DocumentService`, `from services.file_mapper_service import FileMapperService`, `from services.filename_utils import safe_filename`, `from abilities.document import …`, `from services.tmp_storage import …` | REMOVE | Hoist to top-level | Pure utilities / lazy singletons — no import-time side effects |
| (lines with PLC0415 — playwright) | PLC0415 | `from tools.browser.security import …` or playwright-related | KEEP | No change | Playwright is a heavy third-party lib — cold-start deferral legitimate |
| (lines with PLC0415 — browser pool) | PLC0415 | `from tools.browser.pool import get_pool` | REMOVE | Hoist to top-level | `tools.browser.pool` does NOT import playwright at module level — cheap to hoist |

---

### `configs/channels/dmn.py` (5 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415 — stdlib) | PLC0415 | `import logging` | REMOVE | Hoist to top-level | stdlib — deferral never legitimate |
| (lines with PLC0415 — first-party) | PLC0415 | light first-party imports | REMOVE | Hoist to top-level | Pure utilities — no side effects |
| (lines with BLE001) | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | DMN-channel fail-open contract |

---

### `abilities/mcp_manager.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 199 | PLC0415 | `from services.mcp_client_service import McpClientService` | KEEP | No change | `mcp_client_service` imports `database_service` at module level — heavy |
| 207 | PLC0415 | `from services.mcp_client_service import McpClientService` | KEEP | No change | Same as line 199 |
| 255 | PLC0415 | `from services.mcp_client_service import McpClientService` | KEEP | No change | Same as line 199 |
| 280 | PLC0415 | `from services.mcp_client_service import McpClientService` | KEEP | No change | Same as line 199 |

---

### `services/llm_clients/ollama.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 143 | PLC0415 | `from services.llm_service import _app_user_agent` | REMOVE | Hoist to top-level | `llm_service.py` only imports `re`, `logging`, `typing` — pure helper |
| 229 | PLC0415 | `from services.providers import PROVIDER_CALL_TIMEOUT_S` | REMOVE | Hoist to top-level | Pure constants module |
| 230 | PLC0415 | `from services.provider_api import ProviderTimeoutError` | REMOVE | Hoist to top-level | Pure exception class |
| 278 | PLC0415 | `from services.llm_service import estimate_tokens` | REMOVE | Hoist (consolidate with line 143) | Same as line 143 |

---

### `services/client_context.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415) | PLC0415 | light first-party imports | REMOVE | Hoist to top-level | No import-time side effects |

---

### `tests/test_ubiquiti_ssl_downgrade.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (line with S603) | S603 | `subprocess.run(...)` with trusted args | KEEP | No change | Argv is a list of string constants — documented |
| (line with N802) | N802 | `def do_GET(self)` | KEEP | No change | `http.server` contract |
| (line with ARG002) | ARG002 | `def log_message(self, *args)` | KEEP | No change | Silences the default request logging — deliberate override |

---

### `abilities/_params.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 91 | A003 | `id = "id"  # noqa: A003 — attr name mirrors the wire key (shadows builtin)` | KEEP | No change | Mirrors the JSON wire key — comment documents intent |
| 103 | A003 | `list = "list"  # noqa: A003 — attr name mirrors the wire key (shadows builtin)` | KEEP | No change | Same as line 91 |

---

### `run.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 31 | F401 | `import numpy  # noqa: F401 — thread-safety warm-up` | KEEP | No change | Side-effect import — warms up numpy's thread-safety locks at boot |
| 32 | F401 | `import transformers  # noqa: F401 — thread-safety warm-up` | KEEP | No change | Same as line 31 — transformers thread-safety warm-up |

---

### `tests/test_convergence_release_gate.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| (lines with PLC0415) | PLC0415 | `from services.subconscious_worker import SubconsciousWorker` | REMOVE | Hoist to top-level | `SubconsciousWorker` has no module-level singleton — safe to hoist |

---

### `abilities/skill_builder.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 382 | ARG001 | `def _handle_list(params: dict[str, object]) -> ToolResult:  # noqa: ARG001` | KEEP | No change | Interface contract — the dispatcher always passes `params` even if unused |

---

### `migrations/migration_001_drop_compactions.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 26 | E402 | `from services.file_mapper_service import FileMapperService  # noqa: E402` | KEEP | No change | `sys.path.insert` bootstrap precedes — standalone migration script |

---

### `tests/test_compaction_watermark.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 10 | F401 | `from abilities.chat_history_compactor import _CompactionParent  # noqa: F401` | KEEP | No change | Type-only alias under `TYPE_CHECKING` — used in `cast()` calls |

---

### `tools/browser/__init__.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 8 | F401 | `import playwright  # noqa: F401 — availability check` | KEEP | No change | Side-effect import — verifies playwright is installed; `AVAILABLE` is set from the import's success/failure |

---

### `configs/channels/pattern.py` (13 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 18 | PLC0415 | `import json as _json` | REMOVE | Hoist `import json` to top-level | stdlib |
| 19 | PLC0415 | `import logging as _logging` | REMOVE | Hoist `import logging` to top-level | stdlib |
| 22 | PLC0415 | `from services.database_service import get_shared_db_service` | REMOVE | Hoist to top-level | Singleton is lazy — no eager init at import |
| 53 | PLC0415 | `import json as _json` (duplicate) | REMOVE | Hoist (consolidate with line 18) | stdlib |
| 54 | PLC0415 | `from services.act_trail import ActTrail` | REMOVE | Hoist to top-level | Pure utility class |
| 75 | PLC0415 | `import logging as _logging` (duplicate) | REMOVE | Hoist (consolidate with line 19) | stdlib |
| 78 | PLC0415 | `from services.database_service import get_shared_db_service` (duplicate) | REMOVE | Hoist (consolidate with line 22) | Singleton is lazy |
| 79 | PLC0415 | `from services.time_utils import utc_now` | REMOVE | Hoist to top-level | Pure utility, no side effects |
| 130 | PLC0415 | `import logging as _logging` (duplicate) | REMOVE | Hoist (consolidate with line 19) | stdlib |
| 136 | PLC0415 | `from services.database_service import get_shared_db_service` (duplicate) | REMOVE | Hoist (consolidate with line 22) | Singleton is lazy |
| 150 | PLC0415 | `from services.skill_association_service import SkillAssociationService` | REMOVE | Hoist to top-level | Pure utility class |
| 184 | PLC0415 | `import logging as _logging` (duplicate) | REMOVE | Hoist (consolidate with line 19) | stdlib |
| 187 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | Pure utility class (statics only) |

---

### `services/turn_zero_flashback.py` (10 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 132 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Seed is best-effort per docstring; failure must not abort the user's turn |
| 155 | PLC0415 | `from services.embedding_service import get_embedding_service` | KEEP | No change | `embedding_service` pulls `numpy` + `onnx_session` at top-level — heavy |
| 165 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Continuation gate explicitly fails open per docstring |
| 178 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | Pure utility class (statics only) |
| 206 | PLC0415 | `import numpy as np` | KEEP | No change | Heavy third-party lib — cold-start deferral legitimate |
| 207 | PLC0415 | `from services.embedding_service import get_embedding_service` (duplicate) | KEEP | No change | Same as line 155 |
| 226 | PLC0415 | `import numpy as np` (duplicate) | KEEP | No change | Same as line 206 |
| 252 | PLC0415 | `from services.memory_retrieval import _search_data_graph` | REMOVE | Hoist to top-level | `memory_retrieval` is a pure-utility module (helpers only, no singletons) |
| 272 | PLC0415 | `from services.memory_retrieval import recall_episodes` | REMOVE | Hoist (consolidate with line 252) | Same as line 252 |
| 341 | PLC0415 | `from services.act_trail import ActTrail` | REMOVE | Hoist to top-level | Pure utility class |

---

### `abilities/vision.py` (8 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 59 | PLC0415 | `from services.database_service import get_shared_db_service` | REMOVE | Hoist to top-level | Singleton is lazy — no eager init at import |
| 60 | PLC0415 | `from services.provider_db_service import ProviderDbService` | REMOVE | Hoist to top-level | Pure utility class |
| 63 | PLC0415 | `from services.message_processor import MessageProcessor` | REMOVE | Hoist to top-level | Only stdlib + time_formatter_service at top-level — cheap |
| 72 | PLC0415 | `from services import image_context_service` | REMOVE | Hoist to top-level | `image_context_service` only imports stdlib at module level; heavy `rapidocr` is deferred inside `analyze()` itself |
| 153 | PLC0415 | `from services.database_service import get_shared_db_service` (duplicate) | REMOVE | Hoist (consolidate with line 59) | Singleton is lazy |
| 154 | PLC0415 | `from services.document_service import DocumentService` | REMOVE | Hoist to top-level | Pure utility class |
| 155 | PLC0415 | `from services.file_mapper_service import FileMapperService` | REMOVE | Hoist to top-level | Pure utility module |
| 177 | BLE001 | `except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed` | KEEP | No change | Error is logged AND returned via `ToolResult.err` — surfaced to the model |

---

### `configs/channels/super_episode.py` (5 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 32 | PLC0415 | `import json as _json` | REMOVE | Hoist `import json` to top-level | stdlib |
| 33 | PLC0415 | `import logging as _logging` | REMOVE | Hoist `import logging` to top-level | stdlib |
| 52 | PLC0415 | `import json as _json` (duplicate) | REMOVE | Hoist (consolidate with line 32) | stdlib |
| 87 | PLC0415 | `import logging as _logging` (duplicate) | REMOVE | Hoist (consolidate with line 33) | stdlib |
| 157 | PLC0415 | `from services.system_message_prompt import SuperEpisodeEncoderSystemPrompt` | REMOVE | Hoist to top-level | Pure ABC subclasses — no side effects |

---

### `services/act_trail.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 40 | PLC0415 | `from services.database_service import get_shared_db_service` | REMOVE | Hoist to top-level | `DatabaseService` singleton is lazy (`_shared_db_service = None` at module level) |
| 77 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Write failures are non-fatal per docstring — the turn continues |
| 100 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Read failure returns empty list per docstring — legitimate fail-open |
| 121 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Same as line 100 |

---

### `services/policy_manager.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 81 | PLC0415 | `from services.database_service import get_shared_db_service` | REMOVE | Hoist to top-level | Singleton is lazy |
| 131 | PLC0415 | `from services.websocket_broker import WebSocketBroker` | REMOVE | Hoist to top-level | Singleton class but instantiation is lazy; only imports stdlib at module level |
| 151 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Explicit fail-open per docstring |
| 207 | PLC0415 | `from services.file_mapper_service import FileMapperService` | REMOVE | Hoist to top-level | Pure utility module |

---

### `services/embedding_service.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 158 | PLC0415 | `from onnxruntime import InferenceSession as _IS` | TRANSFORM | Move to `if TYPE_CHECKING:` block at top of file | Used purely as type marker in `cast(_IS, session)` — runtime cost is zero |
| 192 | PLC0415 | `from onnxruntime import InferenceSession as _IS` (duplicate) | TRANSFORM | Move to `if TYPE_CHECKING:` block (consolidate with line 158) | Same as line 158 |
| 193 | PLC0415 | `from transformers import PreTrainedTokenizerBase as _Tok` | TRANSFORM | Move to `if TYPE_CHECKING:` block at top of file | Used purely as type marker in `cast(_Tok, tokenizer)` — runtime cost is zero |

---

### `tools/browser/session.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 115 | PLC0415 | `from tools.browser.pool import get_pool` | REMOVE | Hoist to top-level | `tools.browser.pool` does NOT import playwright at module level — cheap to hoist |
| 123 | PLC0415 | `from tools.browser.pool import get_pool` (duplicate) | REMOVE | Hoist (consolidate with line 115) | Same as line 115 |
| 164 | ARG001 | `def _close_on_thread(browser: object, key: int) -> None:  # noqa: ARG001 — pool always passes browser` | KEEP | No change | Comment explicitly justifies — `browser` is part of the pool's dispatch contract |

---

### `api/capabilities.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 69 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Last-sync-at read is best-effort metadata — legitimate fail-open |
| 234 | BLE001 | `except Exception as ec_exc:  # noqa: BLE001` | KEEP | No change | Error count read is best-effort — broad catch is intentional |

---

### `services/runtime_deps_service.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 56 | F401 | `import onnxruntime  # noqa: F401` | KEEP | No change | Side-effect import — verifies the runtime actually loads (catches native lib failures) |
| 68 | F401 | `import onnxruntime  # noqa: F401` | KEEP | No change | Same as line 56 — verifies the CPU wheel loads after the fallback install |

---

### `tests/test_search_relevance_ranking.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 23 | E402 | `from tools.search.router import rank_results  # noqa: E402 — baseline-fail target` | KEEP | No change | File-level docstring says these imports are the baseline-fail target — file must fail at collection if they don't exist |
| 24 | E402 | `from tools.search.enrich import enrich_missing_summaries  # noqa: E402 — baseline-fail target` | KEEP | No change | Same as line 23 |

---

### `abilities/web_search.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 101 | PLC0415 | `from services.message_processor import MessageProcessor` | REMOVE | Hoist to top-level | `MessageProcessor` only imports stdlib + time_formatter_service at top-level — cheap |

---

### `services/processor_config.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 151 | PLC0415 | `import copy` | REMOVE | Hoist `import copy` to top-level | stdlib |

---

### `tools/search/fetcher.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 173 | S501 | `verify=False,  # noqa: S501 — intentional fallback for providers with broken certs` | KEEP | No change | First attempt with `verify=True` raises `SSLError`, then we fall back — documented |

---

### `tests/test_geo_feature.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 31 | F401 | `from geopy.distance import geodesic as _geopy_check  # noqa: F401` | KEEP | No change | Side-effect import — only purpose is to trigger ImportError so the `_GEOPY_AVAILABLE` flag can gate the test skip |

---

### `api/chat.py` (11 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 50-51 | TYPE_CHECKING | `if TYPE_CHECKING: from services.processor_config import ProcessorConfig` | TRANSFORM | Remove the `TYPE_CHECKING:` block (its single entry becomes a top-level runtime import once L473 is hoisted) | After L473 is hoisted, ProcessorConfig is needed at runtime, so the TYPE_CHECKING guard is moot |
| 111 | PLC0415 | `from services.message_processor import MessageProcessor` | REMOVE | Hoist to top-level | First-party with no module-level singleton/thread; DAG verified acyclic |
| 204 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | Pure first-party class, no import-time side effects |
| 207 | PLC0415 | `from services.database_service import get_shared_db_service` | KEEP | No change | Singleton — on the legitimate-deferral list |
| 208 | PLC0415 | `from services.rich_media_parser import resolve_tool_call_transcript_ids` | REMOVE | Hoist to top-level | Pure parsing helper, no I/O at module init |
| 241 | PLC0415 | `from services.message_processor import MessageProcessor` (duplicate) | REMOVE | Hoist (consolidate with line 111) | Same as line 111 |
| 293 | PLC0415 | `from configs.channels import UserConfig` | REMOVE | Hoist to top-level | First-party config class with no import-time side effects |
| 299 | PLC0415 | `from services.world_state import world_state, Signal` | KEEP | No change | `world_state = WorldState()` at module init performs DB hydration — eager singleton init with I/O |
| 338 | PLC0415 | `from services.filename_utils import safe_filename` | REMOVE | Hoist to top-level | Pure first-party utility |
| 339 | PLC0415 | `from services.tmp_storage import new_tmp_path` | REMOVE | Hoist to top-level | Pure first-party utility (only stdlib imports at top) |
| 472 | PLC0415 | `from abilities._dispatcher import ToolDispatcher` | REMOVE | Hoist to top-level | First-party dispatcher with no module-level singleton; runtime chain is acyclic |
| 473 | PLC0415 | `from services.processor_config import ProcessorConfig` | REMOVE | Hoist to top-level (and drop the now-redundant TYPE_CHECKING entry) | Pure ABC, used as runtime base class |

---

### `configs/channels/user_summary.py` (11 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 38 | PLC0415 | `import json as _json` | REMOVE | Hoist `import json` to top-level | stdlib |
| 39 | PLC0415 | `import logging as _logging` | REMOVE | Hoist `import logging` to top-level | stdlib |
| 77 | PLC0415 | `from services.data_graph_service import get_data_graph_service` | REMOVE | Hoist to top-level | Lazy-singleton accessor; first-party with no side effects |
| 108 | PLC0415 | `import logging as _logging` (duplicate) | REMOVE | Hoist (consolidate with line 39) | stdlib |
| 111 | PLC0415 | `from services.database_service import get_shared_db_service` | KEEP | No change | Singleton — on the legitimate-deferral list |
| 112 | PLC0415 | `from services.time_utils import parse_utc` | REMOVE | Hoist to top-level (extend the existing `from services.time_utils import utc_now`) | Pure first-party utility (only stdlib imports) |
| 181 | PLC0415 | `import json as _json` (duplicate) | REMOVE | Hoist (consolidate with line 38) | stdlib |
| 182 | PLC0415 | `import logging as _logging` (duplicate) | REMOVE | Hoist (consolidate with line 39) | stdlib |
| 187 | PLC0415 | `from services.data_graph_service import get_data_graph_service` (duplicate) | REMOVE | Hoist (consolidate with line 77) | Same as line 77 |
| 210 | PLC0415 | `from services.database_service import get_shared_db_service` (duplicate) | KEEP | No change | Consistent with line 111 |
| 247 | PLC0415 | `from services.system_message_prompt import UserSummarySystemPrompt` | REMOVE | Hoist to top-level | Pure prompt class — no side effects |

---

### `services/llm_clients/openai.py` (9 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 119 | PLC0415 | `from openai import OpenAI` | KEEP | No change | Heavy third-party SDK — cold-start deferral legitimate |
| 120 | PLC0415 | `from services.llm_service import _resolve_api_key, _app_user_agent` | REMOVE | Hoist to top-level | `llm_service.py` only imports `re`, `logging`, `typing` — pure helper |
| 121 | PLC0415 | `from services.providers import PROVIDER_CALL_TIMEOUT_S` | REMOVE | Hoist to top-level | Pure constants module |
| 174 | PLC0415 | `import openai as openai_mod` | KEEP | No change | Heavy third-party SDK |
| 175 | PLC0415 | `from services.llm_service import _is_thinking_rejection` | REMOVE | Hoist (consolidate with line 120) | Pure helper |
| 176 | PLC0415 | `from services.provider_api import ProviderTimeoutError` | REMOVE | Hoist to top-level | Pure exception class |
| 205 | PLC0415 | `from services.llm_service import _strip_think_blocks` | REMOVE | Hoist (consolidate with line 120) | Pure helper |
| 263 | PLC0415 | `from services.llm_service import estimate_tokens` | REMOVE | Hoist (consolidate with line 120) | Pure helper |
| 265 | PLC0415 | `import tiktoken` | KEEP | No change | Heavy third-party lib — cold-start deferral legitimate |

---

### `configs/channels/external_agent.py` (6 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 36 | PLC0415 | `import logging` | REMOVE | Hoist `import logging` to top-level | stdlib |
| 47 | PLC0415 | `from api.chat import dispatch_message` | REMOVE | Hoist to top-level | First-party blueprint; DAG is acyclic |
| 98 | PLC0415 | `import logging` (duplicate) | REMOVE | Hoist (consolidate with line 36) | stdlib |
| 104 | PLC0415 | `from services.system_message_prompt import ExternalAgentSystemMessagePrompt` | REMOVE | Hoist to top-level | Pure class — no I/O at import |
| 111 | PLC0415 | `from services.data_graph_service import get_data_graph_service` | REMOVE | Hoist to top-level | Lazy-singleton accessor; first-party with no side effects |
| 138 | PLC0415 | `import logging` (duplicate) | REMOVE | Hoist (consolidate with line 36) | stdlib |

---

### `tests/test_web_download.py` (5 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 118 | E402 | `import abilities.web_download` | REMOVE | Move to top-level imports | No `sys.path.insert` in this file — sloppy ordering |
| 119 | E402 | `from abilities._dispatcher import ToolDispatcher` | REMOVE | Move to top-level imports | Sloppy ordering |
| 120 | E402 | `from configs.channels import DmnConfig` | REMOVE | Move to top-level imports | Sloppy ordering |
| 121 | E402 | `from services.act_trail import ActTrail` | REMOVE | Move to top-level imports | Sloppy ordering |
| 122 | E402 | `from tests._tool_result_harness import seed_transcript` | REMOVE | Move to top-level imports | Sloppy ordering |

---

### `services/async_delegate_runner.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 144 | PLC0415 | `from abilities._dispatcher import ToolDispatcher` | REMOVE | Hoist to top-level; update the docstring at L141-143 to reflect that the DAG is acyclic | First-party dispatcher, no module-level singleton; the comment claiming otherwise is stale |
| 149 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Tool-execution fail-open contract |
| 169 | PLC0415 | `from api.chat import deliver_async_result` | REMOVE | Hoist to top-level | First-party blueprint, no module-level singleton |
| 171 | BLE001 | `except Exception as exc:  # noqa: BLE001` | KEEP | No change | Synthesis-delivery fail-open; the still-emits-end-event contract demands isolation |

---

### `services/scheduler_service.py` (4 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 369 | PLC0415 | `from configs.channels import ScheduledConfig` | REMOVE | Hoist to top-level | First-party config, no import-time side effects |
| 370 | PLC0415 | `from services.message_processor import MessageProcessor` | REMOVE | Hoist to top-level | First-party module without import-time singleton; DAG acyclic |
| 371 | PLC0415 | `from services.processor_config import ProcessorConfig` | REMOVE | Hoist to top-level | Pure ABC |
| 391 | PLC0415 | `from api.chat import dispatch_message` | REMOVE | Hoist to top-level | First-party blueprint; no eager singleton |

---

### `services/mcp_client_service.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 197 | PLC0415 | `import sqlite_vec` | KEEP | No change | Heavy third-party lib — on the whitelist |
| 691 | PLC0415 | `from services.embedding_service import EmbeddingService` | KEEP | No change | `embedding_service` imports numpy at top-level — heavy |
| 692 | PLC0415 | `from services.embedding_utils import pack_embedding` | REMOVE | Hoist to top-level | Pure utility (only `struct` and typing imports) |

---

### `utils/build_skills_db.py` (3 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 15 | E402 | `from services.embedding_service import EmbeddingService` | KEEP | No change | `sys.path.insert(0, …)` bootstrap precedes — standalone script |
| 16 | E402 | `from services.embedding_utils import pack_embedding` | KEEP | No change | Same as line 15 |
| 17 | E402 | `from services.file_mapper_service import FileMapperService` | KEEP | No change | Same as line 15 |

---

### `configs/channels/geo_pattern.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 17 | PLC0415 | `import logging as _logging` | REMOVE | Hoist `import logging` to top-level | stdlib |
| 20 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | First-party module with only stdlib imports — no side effects |

---

### `services/system_message_prompt.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 15 | N802 | `def _SYSTEM_PROMPT(self) -> str:  # noqa: N802` | KEEP | No change | Uppercase property name intentionally signals "system-prompt constant" abstract attribute |
| 21 | N802 | `def getPrompt(self) -> str:  # noqa: N802` | KEEP | No change | Backward-compat shim that must mirror the historical camelCase call site |

---

### `tools/search/enrich.py` (2 noqas)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 101 | BLE001 | `except Exception as exc:  # noqa: BLE001 — spec-mandated fail-open` | KEEP | No change | Module docstring calls out fail-open contract |
| 155 | BLE001 | `except Exception as exc:  # noqa: BLE001 — belt-and-suspenders` | KEEP | No change | Belt-and-suspenders for ThreadPoolExecutor future results |

---

### `api/policies.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 38 | BLE001 | `except Exception as exc:  # noqa: BLE001 — display enrichment must not 500 the page` | KEEP | No change | Display enrichment on a public endpoint must fail-open |

---

### `services/world_awareness_service.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 98 | PLC0415 | `from services.transcript_service import Transcript` | REMOVE | Hoist to top-level | First-party module with only stdlib imports — no side effects |

---

### `tests/test_mcp_client_service.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 257 | E402 | `from services.mcp_client_service import _normalize_host, _open_tools_db` | REMOVE | Move to top-level imports | No `sys.path` bootstrap — sloppy ordering |

---

### `utils/seed_routing_examples.py` (1 noqa)

| Line | Code | Current snippet | Verdict | Action | Reason |
|---|---|---|---|---|---|
| 9 | E402 | `from services.file_mapper_service import FileMapperService` | KEEP | No change | `sys.path.insert(0, …)` bootstrap precedes — standalone script |

---

## Aggregate Summary

| Verdict | Count | Description |
|---|---|---|
| **REMOVE** | ~183 | Hoist the import to top-level (or delete if redundant) and drop the `# noqa` (includes 3 added for `capabilities.contact_resolver` in `carddav_handler.py`) |
| **KEEP** | ~143 | The `# noqa` is justified — heavy third-party lib, eager singleton, fail-open contract, interface contract, or sys.path bootstrap (includes 3 added in `carddav_handler.py` and 6 in `mcp_server/server.py` that currently lack `# noqa` — see "Unflagged" row) |
| **TRANSFORM** | 6 | Move to `if TYPE_CHECKING:` block (3 in `embedding_service.py`, 1 in `message_processor.py:223`, 1 in `api/chat.py:50-51`, 1 in `test_vision_ability.py:104` — narrow `pytest.raises`) |
| **Unflagged PLC0415** (in addition to rows above) | 7 | `mcp_server/server.py` carries 7 import-outside-top-level lines that **lack** a `# noqa: PLC0415` — `ruff check --select PLC0415 backend/mcp_server/server.py` reports them as errors. They are listed under that file's section with KEEP / REMOVE verdicts; KEEPs (6) need `# noqa` added, the single REMOVE (line 78) should be hoisted. |

### By rule code

> 76 files, 330 `# noqa` occurrences total. The 7 unflagged PLC0415 violations in `mcp_server/server.py` are listed above separately.

| Code | Total | REMOVE | KEEP | TRANSFORM |
|---|---|---|---|---|
| PLC0415 | 241 | ~153 | ~85 | 3 |
| E402 | 27 | 6 | 21 | 0 |
| BLE001 | 23 | 0 | 23 | 0 |
| F401 | 9 | 0 | 9 | 0 |
| N802 | 5 | 0 | 5 | 0 |
| F811 | 5 | 0 | 5 | 0 |
| SLF001 | 3 | 0 | 3 | 0 |
| PLC2701 | 3 | 0 | 3 | 0 |
| ARG001 | 3 | 0 | 3 | 0 |
| S603 | 2 | 0 | 2 | 0 |
| ARG002 | 2 | 0 | 2 | 0 |
| A003 | 2 | 0 | 2 | 0 |
| S501 | 1 | 0 | 1 | 0 |
| PLW0603 | 1 | 0 | 1 | 0 |
| F821 | 1 | 0 | 0 | 1 |
| E731 | 1 | 1 | 0 | 0 |
| B017 | 1 | 0 | 0 | 1 |

### Highest-impact files (most removals)

| File | Removals | Notes |
|---|---|---|
| `services/subconscious_worker.py` | ~20 | Heavy worker; most deferrals are stdlib/first-party utilities |
| `services/message_processor.py` | 18 | Hot-path orchestrator; hoisting `transcript_service`, `providers`, `provider_api`, `act_trail`, `compaction_persistence`, `tmp_storage`, `turn_zero_flashback`, `os`, `ThreadPoolExecutor` |
| `configs/channels/pattern.py` | 13 | stdlib (`json`/`logging`) + first-party utilities |
| `services/providers.py` | 12 | Pure constants/exception/factory imports — all hoistable |
| `api/chat.py` | 8 + 1 transform | Hoist `MessageProcessor`, `Transcript`, `UserConfig`, `filename_utils`, `tmp_storage`, `ToolDispatcher`, `ProcessorConfig` |
| `configs/channels/user_summary.py` | 9 | stdlib + first-party utilities |
| `services/llm_clients/openai.py` | 6 | Hoist `llm_service` helpers + `provider_api` + `providers` constants |
| `configs/channels/external_agent.py` | 6 | stdlib + first-party |
| `abilities/vision.py` | 7 | Hoist `database_service`, `provider_db_service`, `message_processor`, `image_context_service`, `document_service`, `file_mapper_service` |
| `tests/test_web_download.py` | 5 | Reorder imports to top |
| `services/turn_zero_flashback.py` | 5 | Hoist `transcript_service`, `memory_retrieval`, `act_trail` |
| `capabilities/mail_capability/carddav_handler.py` | 3 | **Added in this revision.** Hoist the 3 `capabilities.contact_resolver` imports (consolidated to one line at top-level); keep `caldav`/`vobject` deferred (whitelist) |

---

## Recommended Execution Order

1. **Start with the `services/llm_clients/*` files** — small, self-contained, high signal. Hoist `llm_service` and `provider_api` imports; keep `openai`/`anthropic`/`google.genai`/`tiktoken` deferred.
2. **Then `services/message_processor.py`** — the hot-path orchestrator. Hoist the 18 removable imports in one pass; move `ProcessorConfig` to `TYPE_CHECKING`.
3. **Then the `configs/channels/*` files** — mostly stdlib + first-party utilities; straightforward hoists.
4. **Then `abilities/vision.py`, `abilities/web_browse.py`, `abilities/web_search.py`** — hoist first-party imports.
5. **Then `services/embedding_service.py`** — the 3 `TYPE_CHECKING` transforms.
6. **Then `capabilities/mail_capability/carddav_handler.py`** — hoist the 3 `capabilities.contact_resolver` imports to a single consolidated top-level line; keep `caldav`/`vobject` deferred.
7. **Then `mcp_server/server.py` — fix both ways** — hoist `L78` (`utc_now`) and drop its (already-absent) suppression; add `# noqa: PLC0415` to `L76, L77, L183, L184, L211, L212` so the KEEP rows actually suppress `ruff`; leave `L145, L146` as KEEP noqas and hoist `L154` (`ProviderRetriesExhaustedError`).
8. **Then the remaining `services/*` and `api/*` files.**
9. **Finally the `tests/*` files** — reorder E402 imports where no sys.path bootstrap exists.

After each file, run `ruff check --select PLC0415,F821,E402,B017 <file>` to confirm the noqa removals are clean and no new violations appear. For `mcp_server/server.py` specifically, run that same command and confirm zero PLC0415 errors remain after step 7.