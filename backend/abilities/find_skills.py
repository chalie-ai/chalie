"""find_skills — surface curated step-by-step skill playbooks.

Queries skills.sqlite through the discovery cascade in
:class:`abilities._search.SearchableAbility`: a ``query`` array of intents,
each run exact-title → bm25 on the title only (segment-gated) → vector on the
full prose. Matching playbooks are returned with any personalisation rules
derived from the user's behavioural patterns (written by SkillAssociationService).

What it returns is shaped by what a skill IS — a playbook: the row carries the
full ``content`` the model executes, not a tool activation.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* matches → a JSON list, one row per skill ``{"name": <title>, "content": <full
  playbook content>, "rules": [{"pattern_name", "rule"}, …]}`` with ``meta
  count=<n>``.
* zero hits against a HEALTHY index → SUCCESS with ``count=0`` and a body
  ``{"matches": [], "note": …}`` — never an error.
* a missing/blank query → ``err(code="missing-params")`` naming ``query``.
* the index file missing OR a sqlite error while probing it →
  ``err(code="skill-index-error")``. This is the loud guardrail: a corrupt or
  unreadable index must NEVER masquerade as "no skills found". ``_bm25_name`` /
  ``_vector_name`` swallow a search-time failure on :class:`models.skill.Skill`'s
  ``search_by_title_bm25`` / ``search_by_vector`` to ``[]``, so this ability
  probes the index FIRST — past a successful probe, an empty result means
  genuinely-no-matches, not a broken index.
"""

import logging
import sqlite3
from typing import ClassVar, cast

from configs.enums.param_key import Keys
from abilities._result import ToolResult
from abilities._search import KNN_DEPTH, SearchableAbility
from contracts.params.find_skills_params_bag import FindSkillsParamsBag
from contracts.params.param_bag import ParamBag
from models.skill import Skill
from models.skill_association import SkillAssociation
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_BROADEN_HINT = "Try broadening the query or describing the task differently."


class FindSkillsAbility(SearchableAbility):
    PARAMS: ClassVar[type[ParamBag] | None] = FindSkillsParamsBag
    NAME: ClassVar[str] = "find_skills"

    DISCOVERABLE: ClassVar[bool] = False  # framework discovery tool; always pinned, never discovered


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
                "type": "array",
                "items": {"type": "string"},
                "description": "One skill name or one described task per entry.",
            },
        },
        "required": [Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _LOG_PREFIX = "[FIND_SKILLS]"

    def run(self, params: FindSkillsParamsBag) -> ToolResult:
        rows = params.query

        db_path = FileMapperService.get_skills_db_path()
        if not db_path.exists():
            logger.warning(f"{self._LOG_PREFIX} skills.sqlite not found at {db_path}")
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

        title_index = self._title_index()
        pins, disc, _not_found = self._discover(
            rows,
            lambda candidate: (title_index.get(candidate), False),
            self._bm25_name,
            self._vector_name,
        )
        return self._result(cast("list[int]", pins + disc))

    def _probe_index(self) -> "ToolResult | None":
        try:
            Skill.all().limit(1).select("id")
            Skill.probe_search_fts()
        except sqlite3.Error as exc:
            logger.warning(f"{self._LOG_PREFIX} index probe failed: {exc}")
            return ToolResult.err(
                "The skill index could not be read.",
                code="skill-index-error",
                hint="the skills database is corrupt or unreadable.",
            )
        return None

    def _title_index(self) -> dict[str, int]:
        """``{normalised title: skill_id}`` for the exact-name rung."""
        rows = Skill.filter("enabled", 1).select("id", "title")
        return {
            self._norm(cast("str", row["title"])): cast("int", row["id"])
            for row in rows
        }

    def _bm25_name(self, terms: list[str]) -> list[object]:
        """Rung 2: bm25 over the skill TITLE only, gated by title-segment alignment."""
        fts = self._fts_or(terms)
        if not fts:
            return []
        try:
            rows = Skill.search_by_title_bm25(fts)
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} FTS search failed (skipping): {exc}")
            return []
        return self._gate(rows, terms)

    def _vector_name(self, terms: list[str]) -> list[object]:
        """Rung 3: vector search on the full prose, per-title deduped, kept only
        at/under ``VEC_CEILING``."""
        blob = self._embed(terms)
        if blob is None:
            return []
        try:
            rows = Skill.search_by_vector(blob, KNN_DEPTH)
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} vec search failed (skipping): {exc}")
            return []
        return self._ceiling(rows)

    def _result(self, ids: list[int]) -> ToolResult:
        """Build the success ToolResult from the cascade's winning skill ids.

        Zero hits past a healthy probe is a genuine no-match SUCCESS, never an error."""
        if not ids:
            return ToolResult.ok(
                {"matches": [], "note": f"No skill playbooks found. {_BROADEN_HINT}"},
                count=0,
            )
        matches = [self._row(skill_id) for skill_id in ids]
        return ToolResult.ok(matches, count=len(matches))

    def _row(self, skill_id: int) -> dict[str, object]:
        title, content, rules = self._fetch_detail(skill_id)
        return {"name": title, "content": content, "rules": rules}

    def _fetch_detail(self, skill_id: int) -> tuple[str, str, list[dict[str, object]]]:
        skill = Skill.get(skill_id)
        title, content = (skill.title, skill.content) if skill is not None else ("", "")

        rules = SkillAssociation.filter("skill_id", skill_id).select("pattern_name", "rule")

        return title, content, rules
