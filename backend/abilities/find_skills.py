"""find_skills — surface curated step-by-step skill playbooks.

Queries skills.sqlite with the shared keyword (trigram FTS5) + vector search
and returns matching skill playbooks together with any personalisation rules
derived from the user's behavioural patterns (written by SkillAssociationService).

The ``query`` is the keyword grammar described in ``_search.build_keyword_query``
(``+required``, ``-excluded``, bare optional), not prose.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* matches → a JSON list, one row per skill ``{"name": <title>, "source":
  <"keyword"|"vector">, "content": <full playbook content>, "rules":
  [{"pattern_name", "rule"}, …]}`` with ``meta count=<n>``. The content field is
  the actionable payload — it IS the playbook the model executes.
* zero hits against a HEALTHY index → SUCCESS with ``count=0`` and a body
  ``{"matches": [], "note": "No skill playbooks found … broaden …"}`` — never an
  error.
* a missing/blank query → ``err(code="missing-params")`` naming ``query``.
* the index file missing OR a sqlite error while probing it →
  ``err(code="skill-index-error")``. This is the loud guardrail: a corrupt or
  unreadable index must NEVER masquerade as "no skills found".
* an embedding failure leaves the keyword signal alone and tags the success
  ``meta degraded=true`` — never silently fewer signals.

The shared ``_vec_search`` / ``_fts_search`` helpers in ``_search.py`` swallow
sqlite failures to ``[]`` (a behaviour find_tools depends on). This ability
therefore probes the index FIRST — past a successful probe, an empty result
means genuinely-no-matches, not a broken index.
"""

import logging
import sqlite3
from pathlib import Path
from typing import ClassVar, cast

from abilities._params import Keys
from abilities._result import ToolResult
from abilities._search import KNN_DEPTH, SearchableAbility, build_keyword_query
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 3
_MAX_LIMIT = 5
_BROADEN_HINT = "Try broadening the query or describing the task differently."


class FindSkillsAbility(SearchableAbility):
    DISCOVERABLE: ClassVar[bool] = False  # framework discovery tool; always pinned, never discovered

    def get_name(self) -> str:
        return "find_skills"

    def get_summary(self) -> str:
        return (
            "Find a step-by-step skill playbook for a complex task like research, "
            "planning, or analysis."
        )

    def get_examples(self) -> list[str]:
        return [
            "help me structure a competitor analysis report",
            "what's the best framework for planning meals for the week",
            "I want to write a great performance review for my team",
            "walk me through how to plan a fitness routine",
            "give me a structured approach to managing a complex project",
            "help me create a research framework for evaluating new technologies",
        ]

    def get_search_tooltip(self) -> str:
        return "discover procedural skill playbooks for complex tasks"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "string",
                "description": (
                    "Keywords describing the task — NOT a sentence. Prefix a term "
                    "with + to require it, - to exclude it; a bare term is optional. "
                    "Example: +research +competitor"
                ),
            },
            Keys.limit: {
                "type": "integer",
                "description": "Max results (default 3, max 5).",
            },
        },
        "required": [Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _DB_PATH: ClassVar[Path] = FileMapperService.get_skills_db_path()
    _LOG_PREFIX = "[FIND_SKILLS]"

    def run(self, params: dict[str, object]) -> ToolResult:
        query = cast("str", params.get(Keys.query, "")).strip()
        limit = min(cast("int", params.get(Keys.limit, _DEFAULT_LIMIT)) or _DEFAULT_LIMIT, _MAX_LIMIT)
        logger.info(f"{self._LOG_PREFIX} query='{query}' limit={limit}")

        if not query:
            return ToolResult.err(
                "A non-empty 'query' is required.",
                code="missing-params",
                hint="describe the task or topic you need a playbook for.",
                valid=("query",),
            )

        if not self._DB_PATH.exists():
            logger.warning(f"{self._LOG_PREFIX} skills.sqlite not found at {self._DB_PATH}")
            return ToolResult.err(
                "The skill index is unavailable.",
                code="skill-index-error",
                hint="the skills database is missing; try again later.",
            )

        # Probe the index BEFORE searching. The shared search helpers swallow
        # sqlite errors to [] — without this probe a corrupt/unreadable index
        # would render as a zero-hit SUCCESS. A probe failure is loud instead.
        if (probe_err := self._probe_index()) is not None:
            return probe_err

        fts_match, embed_text = build_keyword_query(query)

        blob: bytes | None = None
        degraded = False
        if embed_text:
            try:
                from services.embedding_service import EmbeddingService  # noqa: PLC0415
                blob = cast("bytes", pack_embedding(EmbeddingService().generate_embedding(embed_text, mp=self.mp)))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{self._LOG_PREFIX} embedding failed, keyword-only: {exc}")
                degraded = True

        rows = self._search(fts_match, blob, limit)
        return self._result(query, rows, degraded=degraded)

    def _probe_index(self) -> "ToolResult | None":
        try:
            conn = sqlite3.connect(str(self._DB_PATH))
            try:
                conn.execute("SELECT 1 FROM skills LIMIT 1").fetchone()
                conn.execute(
                    "SELECT 1 FROM skill_search_fts "
                    "WHERE skill_search_fts MATCH 'probe' LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(f"{self._LOG_PREFIX} index probe failed: {exc}")
            return ToolResult.err(
                "The skill index could not be read.",
                code="skill-index-error",
                hint="the skills database is corrupt or unreadable.",
            )
        return None

    def _result(self, query: str, rows: list[dict[str, object]], *, degraded: bool = False) -> ToolResult:
        """Build the success ToolResult from the combined keyword+vector rows.

        Zero hits past a healthy probe is a genuine no-match SUCCESS, never an
        error. *degraded* tags a run whose embedding signal was unavailable.
        """
        if not rows:
            body = {
                "matches": [],
                "note": f'No skill playbooks found for "{query}". {_BROADEN_HINT}',
            }
            if degraded:
                return ToolResult.ok(body, query=query, count=0, degraded=True)
            return ToolResult.ok(body, query=query, count=0)

        skill_rows = [self._row(r) for r in rows]
        if degraded:
            return ToolResult.ok(skill_rows, query=query, count=len(skill_rows), degraded=True)
        return ToolResult.ok(skill_rows, query=query, count=len(skill_rows))

    def _row(self, merged: dict[str, object]) -> dict[str, object]:
        content, rules = self._fetch_detail(cast("int", merged["key"]))
        return {
            "name": merged["label"],
            "source": merged["source"],
            "content": content,
            "rules": rules,
        }

    def _search(self, fts_match: str, blob: bytes | None, limit: int) -> list[dict[str, object]]:
        vec_rows = self._vec_search(
            blob,
            """
                SELECT s.id, s.title, v.distance
                FROM skill_search_vec v
                JOIN skill_search_entries e ON e.id = v.rowid
                JOIN skills s ON s.id = e.skill_id
                WHERE v.embedding MATCH ? AND k = ?
                AND s.enabled = 1
                ORDER BY v.distance ASC
            """,
            (blob, KNN_DEPTH),
        )
        fts_rows = self._fts_search(
            fts_match,
            """
                SELECT s.id, s.title, bm25(skill_search_fts) AS score
                FROM skill_search_fts
                JOIN skill_search_entries e ON e.id = skill_search_fts.rowid
                JOIN skills s ON s.id = e.skill_id
                WHERE skill_search_fts MATCH ?
                AND s.enabled = 1
                ORDER BY score ASC
            """,
            (fts_match,),
        )
        return self._combine(vec_rows, fts_rows, limit)

    def _fetch_detail(self, skill_id: int) -> tuple[str, list[dict[str, object]]]:
        conn = sqlite3.connect(str(self._DB_PATH))
        try:
            row = conn.execute(
                "SELECT content FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
            content = row[0] if row else ""

            rules = conn.execute(
                "SELECT pattern_name, rule FROM skill_associations WHERE skill_id = ?",
                (skill_id,),
            ).fetchall()
        finally:
            conn.close()

        return content, [{"pattern_name": r[0], "rule": r[1]} for r in rules]
