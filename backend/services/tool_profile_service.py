"""
Tool Profile Service — Builds, stores, queries, and enriches tool capability profiles.

Profiles are LLM-generated structured descriptions of what each tool/skill does,
when to use it, and example usage scenarios. Used to inject rich capability
context into LLM prompts for skill/tool discovery.

Profiles are stored in tool_capability_profiles SQLite table with:
- short_summary: one-sentence description for triage prompt injection
- full_profile: detailed description for ACT prompt injection
- embedding: stored in tool_capability_profiles_vec virtual table for cosine similarity

Bootstrap: called on startup to build profiles for any missing tool/skill.
Enrichment: triggered by high-salience episodes or idle-time background service.
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from typing import Optional

from services.embedding_utils import pack_embedding as _pack_embedding
from services.innate_skills.registry import ALL_SKILL_NAMES

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TOOL PROFILE]"

# MemoryStore cache key and TTL
TRIAGE_SUMMARIES_CACHE_KEY = "tool_triage_summaries"
TRIAGE_SUMMARIES_TTL = 300  # 5 minutes

# Bump to force all profile embeddings to regenerate.
# v2: embed short_summary + keywords only (not full_profile).
# v3: force re-seed to write keywords column.
_EMBEDDING_VERSION = 3


def _compute_manifest_hash(manifest: dict) -> str:
    """MD5 hash of manifest + embedding version for staleness detection."""
    content = json.dumps(manifest, sort_keys=True) + f":emb_v{_EMBEDDING_VERSION}"
    return hashlib.md5(content.encode()).hexdigest()


def _truncate_keywords(keywords: str, max_len: int = 256) -> str:
    """Truncate keywords to max_len, cleanly at last complete keyword."""
    if len(keywords) <= max_len:
        return keywords
    parts = keywords.split(',')
    result = []
    length = 0
    for part in parts:
        part = part.strip()
        added_len = len(part) + (1 if result else 0)
        if length + added_len > max_len:
            break
        result.append(part)
        length += added_len
    return ','.join(result)


def _read_tool_source(tool_name: str, max_lines: int = 3000) -> str:
    """Read all source files from a tool directory for profile enrichment.

    Language-agnostic: includes every text-based source file in the tool
    directory. The LLM decides what is capability-relevant vs boilerplate.
    Binary files and build artifacts are excluded by extension.
    """
    from pathlib import Path

    tools_dir = Path(__file__).parent.parent / "tools" / tool_name
    if not tools_dir.is_dir():
        return ""

    # Skip binary / build artifacts — everything else is fair game
    skip_ext = {".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe",
                ".wasm", ".jar", ".zip", ".tar", ".gz", ".png", ".jpg",
                ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf"}
    skip_dirs = {"__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build"}

    parts = []
    total_lines = 0

    for source_file in sorted(tools_dir.rglob("*")):
        if not source_file.is_file():
            continue
        if source_file.suffix.lower() in skip_ext:
            continue
        if any(d in source_file.parts for d in skip_dirs):
            continue
        try:
            source = source_file.read_text(encoding="utf-8", errors="replace")
            file_lines = source.splitlines()
            rel_path = source_file.relative_to(tools_dir)
            if total_lines + len(file_lines) > max_lines:
                remaining = max_lines - total_lines
                if remaining > 0:
                    parts.append(f"-- {rel_path} (truncated) --\n" + "\n".join(file_lines[:remaining]))
                    total_lines += remaining
                break
            parts.append(f"-- {rel_path} --\n{source}")
            total_lines += len(file_lines)
        except Exception:
            continue

    return "\n\n".join(parts)


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM response, tolerating markdown fences and preamble."""
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response (len={len(text)})")
    return json.loads(text[start:end + 1])


class ToolProfileService:
    """Builds, stores, queries, and enriches tool capability profiles."""

    def __init__(self, db_service=None):
        self._db = db_service

    def _get_db(self):
        if self._db:
            return self._db
        from services.database_service import get_shared_db_service
        return get_shared_db_service()

    def _get_llm(self):
        from services.llm_service import create_llm_service
        from services.config_service import ConfigService
        agent_cfg = ConfigService.resolve_agent_config('cognitive-triage')
        return create_llm_service(agent_cfg)

    def _get_embedding_service(self):
        from services.embedding_service import EmbeddingService
        return EmbeddingService()

    def _get_store(self):
        from services.memory_client import MemoryClientService
        return MemoryClientService.create_connection(decode_responses=True)

    # -- Profile Building ------------------------------------------------------

    def build_profile(self, tool_name: str, manifest: dict, force: bool = False) -> dict:
        """Build and store a capability profile for an external tool."""
        logger.info(f"{LOG_PREFIX} Building profile for tool: {tool_name}")

        description = manifest.get('documentation') or manifest.get('description', tool_name)
        manifest_hash = _compute_manifest_hash(manifest)

        # Check if profile is current (skip when caller has already decided a rebuild is needed)
        if not force and not self.check_staleness(tool_name, manifest_hash):
            logger.info(f"{LOG_PREFIX} Profile for {tool_name} is current, skipping")
            return self.get_full_profile(tool_name) or {}

        # Query related episodes for enrichment context
        episodes_text = self._get_related_episodes(description)

        # Read tool source code for capability inference
        source_code = _read_tool_source(tool_name)

        # Build LLM prompt
        prompt_template = self._load_prompt('tool-profile-builder')
        prompt = (
            prompt_template
            .replace('{{manifest}}', json.dumps(manifest, indent=2))
            .replace('{{episodes}}', episodes_text)
            .replace('{{source_code}}', source_code or '(source not available)')
        )

        try:
            llm = self._get_llm()
            response_text = llm.send_message("", prompt).text
            profile_data = _extract_json(response_text)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM profile build failed for {tool_name}: {e}")
            profile_data = self._fallback_profile(tool_name, manifest)

        anti_scenarios = profile_data.get('anti_scenarios', [])[:20]

        # Generate embedding from short_summary + keywords
        embedding = None
        keywords = _truncate_keywords(profile_data.get('keywords', ''))
        try:
            emb_service = self._get_embedding_service()
            short_summary = profile_data.get('short_summary', '')
            embedding_text = f"{short_summary} {keywords}" if keywords else short_summary
            embedding = emb_service.generate_embedding(embedding_text)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding generation failed for {tool_name}: {e}")

        # Upsert into database
        db = self._get_db()
        try:
            with db.connection() as conn:
                cursor = conn.cursor()
                effort_tier = profile_data.get('effort_tier', 'moderate')
                if effort_tier not in ('trivial', 'light', 'moderate', 'deep'):
                    effort_tier = 'moderate'

                descriptor = profile_data.get('descriptor', f'{tool_name}')

                cursor.execute(
                    """
                    INSERT INTO tool_capability_profiles
                        (tool_name, tool_type, short_summary, full_profile,
                         anti_scenarios, complementary_skills, manifest_hash, domain,
                         effort, descriptor, keywords, updated_at)
                    VALUES (?, 'tool', ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT (tool_name) DO UPDATE SET
                        tool_type = 'tool',
                        short_summary = EXCLUDED.short_summary,
                        full_profile = EXCLUDED.full_profile,
                        anti_scenarios = EXCLUDED.anti_scenarios,
                        complementary_skills = EXCLUDED.complementary_skills,
                        manifest_hash = EXCLUDED.manifest_hash,
                        domain = EXCLUDED.domain,
                        effort = EXCLUDED.effort,
                        descriptor = EXCLUDED.descriptor,
                        keywords = EXCLUDED.keywords,
                        updated_at = datetime('now')
                    """,
                    (
                        tool_name,
                        profile_data.get('short_summary', f'{tool_name} tool')[:100],
                        profile_data.get('full_profile', description),
                        json.dumps(anti_scenarios),
                        json.dumps(profile_data.get('complementary_skills', [])),
                        manifest_hash,
                        profile_data.get('domain', 'Other'),
                        effort_tier,
                        descriptor,
                        keywords,
                    )
                )

                # Store embedding in vec table
                if embedding is not None:
                    row = cursor.execute(
                        "SELECT rowid FROM tool_capability_profiles WHERE tool_name = ?",
                        (tool_name,)
                    ).fetchone()
                    if row:
                        blob = _pack_embedding(embedding)
                        cursor.execute(
                            "DELETE FROM tool_capability_profiles_vec WHERE rowid = ?",
                            (row[0],)
                        )
                        cursor.execute(
                            "INSERT INTO tool_capability_profiles_vec(rowid, embedding) VALUES (?, ?)",
                            (row[0], blob)
                        )

                cursor.close()
            logger.info(f"{LOG_PREFIX} Upserted profile for {tool_name}")
        except Exception as e:
            logger.error(f"{LOG_PREFIX} DB upsert failed for {tool_name}: {e}")
        finally:
            if not self._db:
                db.close_pool()

        # Invalidate triage summaries cache
        self._invalidate_cache()

        return profile_data


    # -- Enrichment ------------------------------------------------------------

    # -- Query -----------------------------------------------------------------

    def get_triage_summaries(self) -> str:
        """
        Get pre-formatted tool summaries grouped by domain for triage prompt injection.
        Cached in MemoryStore for 5 minutes.
        """
        try:
            ms = self._get_store()
            cached = ms.get(TRIAGE_SUMMARIES_CACHE_KEY)
            if cached:
                return cached
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Triage summaries cache read failed (non-fatal): {e}", exc_info=False)

        summaries = self._build_triage_summaries()

        try:
            ms = self._get_store()
            ms.setex(TRIAGE_SUMMARIES_CACHE_KEY, TRIAGE_SUMMARIES_TTL, summaries)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Triage summaries cache write failed (non-fatal): {e}", exc_info=False)

        return summaries

    def _build_triage_summaries(self) -> str:
        """Build domain-grouped tool summaries from profiles table (DB-driven, tool-agnostic).

        Falls back to manifest-derived summaries when the DB has no tool rows
        (e.g., fresh install before LLM profile bootstrap completes).
        """
        db = self._get_db()
        rows = None
        try:
            rows = db.fetch_all(
                "SELECT tool_name, tool_type, short_summary, domain, effort "
                "FROM tool_capability_profiles ORDER BY domain, tool_name"
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to fetch profiles: {e}")
        finally:
            if not self._db:
                db.close_pool()

        # Filter to only external tools (skills are always available, not listed in triage)
        by_domain = defaultdict(list)
        if rows:
            for r in rows:
                if r['tool_type'] == 'tool':
                    domain = r.get('domain') or 'Other'
                    summary = r['short_summary']
                    effort = r.get('effort') or 'moderate'
                    summary += f" (effort: {effort})"
                    by_domain[domain].append(f"- {r['tool_name']}: {summary}")

        # Fallback: no tool rows in DB -> build from manifests directly
        if not by_domain:
            return self._manifest_fallback_summaries()

        # Coverage check: any on-demand tools missing from DB profiles?
        # Merge manifest-derived summaries for tools not yet profiled.
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            on_demand = set(registry.get_on_demand_tools())
            profiled = {r['tool_name'] for r in (rows or []) if r['tool_type'] == 'tool'}
            missing = on_demand - profiled
            if missing:
                for tool_name in sorted(missing):
                    tool_data = registry.tools.get(tool_name)
                    if not tool_data:
                        continue
                    fallback = self._fallback_profile(tool_name, tool_data['manifest'])
                    domain = fallback.get('domain', 'Other')
                    summary = fallback['short_summary'] + " (effort: moderate)"
                    by_domain[domain].append(f"- {tool_name}: {summary}")
                logger.info(f"{LOG_PREFIX} Added manifest fallback for {len(missing)} unprofiled tool(s): {sorted(missing)}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Coverage check failed: {e}")

        lines = []
        for domain in sorted(by_domain.keys()):
            lines.append(f"## {domain}")
            lines.extend(by_domain[domain])
            lines.append("")

        return "\n".join(lines).strip()

    def get_full_profile(self, tool_name: str) -> Optional[dict]:
        """Get full profile row from database."""
        db = self._get_db()
        try:
            rows = db.fetch_all(
                "SELECT * FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            )
            if rows:
                row = dict(rows[0])
                # Parse JSON fields
                for field in ('anti_scenarios', 'complementary_skills'):
                    if isinstance(row.get(field), str):
                        row[field] = json.loads(row[field])
                return row
            return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} get_full_profile failed for {tool_name}: {e}")
            return None
        finally:
            if not self._db:
                db.close_pool()

    def get_profiles_for_tools(self, tool_names: list) -> list:
        """Batch fetch full_profile for a list of tools (for ACT prompt injection)."""
        if not tool_names:
            return []
        db = self._get_db()
        try:
            placeholders = ','.join(['?'] * len(tool_names))
            rows = db.fetch_all(
                f"SELECT tool_name, short_summary, full_profile FROM tool_capability_profiles WHERE tool_name IN ({placeholders})",
                tuple(tool_names)
            )
            return [dict(r) for r in rows] if rows else []
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} get_profiles_for_tools failed: {e}")
            return []
        finally:
            if not self._db:
                db.close_pool()

    def check_staleness(self, tool_name: str, current_hash: str = None) -> bool:
        """Return True if profile needs rebuilding (missing or stale)."""
        db = self._get_db()
        try:
            rows = db.fetch_all(
                "SELECT manifest_hash FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            )
            if not rows:
                return True  # No profile exists
            if current_hash and rows[0]['manifest_hash'] != current_hash:
                return True  # Manifest changed
            return False
        except Exception:
            return True
        finally:
            if not self._db:
                db.close_pool()

    def rebuild_if_stale(self, tool_name: str) -> bool:
        """Rebuild profile if manifest has changed. Returns True if rebuilt."""
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            if tool_name not in registry.tools:
                return False
            manifest = registry.tools[tool_name]['manifest']
            current_hash = _compute_manifest_hash(manifest)
            if self.check_staleness(tool_name, current_hash):
                self.build_profile(tool_name, manifest)
                return True
            return False
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} rebuild_if_stale failed for {tool_name}: {e}")
            return False

    def seed_builtin_profiles(self) -> None:
        """
        Seed hardcoded profiles for first-party built-in tools.

        Reads BUILTIN_TOOL_PROFILES from tool_library_service and upserts each
        entry into tool_capability_profiles + tool_capability_profiles_vec using
        the same manifest_hash as build_profile() computes. The existing staleness
        gate in bootstrap_all() will therefore naturally skip these tools — their
        hash matches so check_staleness() returns False.
        """
        from services.tool_library_service import BUILTIN_TOOL_PROFILES, TOOL_METADATA

        seeded = 0
        for tool_name, profile in BUILTIN_TOOL_PROFILES.items():
            try:
                manifest = TOOL_METADATA.get(tool_name)
                if not manifest:
                    logger.debug(f"{LOG_PREFIX} seed_builtin_profiles: no TOOL_METADATA for {tool_name}, skipping")
                    continue

                manifest_hash = _compute_manifest_hash(manifest)

                # Skip if already seeded with current hash
                if not self.check_staleness(tool_name, manifest_hash):
                    logger.debug(f"{LOG_PREFIX} seed_builtin_profiles: {tool_name} is current, skipping")
                    continue

                short_summary = profile["short_summary"]
                full_profile = profile["full_profile"]
                effort = profile.get("effort", "moderate")
                domain = profile.get("domain", "Other")
                descriptor = profile.get("descriptor", tool_name)
                keywords = _truncate_keywords(profile.get("keywords", ""))

                # Build embedding from short_summary + keywords
                embedding = None
                try:
                    emb_service = self._get_embedding_service()
                    embedding_text = f"{short_summary} {keywords}" if keywords else short_summary
                    embedding = emb_service.generate_embedding(embedding_text)
                except Exception as e:
                    logger.warning(f"{LOG_PREFIX} seed_builtin_profiles: embedding failed for {tool_name}: {e}")

                # Upsert profile
                db = None
                db = self._get_db()
                try:
                    with db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            INSERT INTO tool_capability_profiles
                                (tool_name, tool_type, short_summary, full_profile,
                                 anti_scenarios, complementary_skills, manifest_hash, domain,
                                 effort, descriptor, keywords, updated_at)
                            VALUES (?, 'tool', ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                            ON CONFLICT (tool_name) DO UPDATE SET
                                tool_type = 'tool',
                                short_summary = EXCLUDED.short_summary,
                                full_profile = EXCLUDED.full_profile,
                                anti_scenarios = EXCLUDED.anti_scenarios,
                                complementary_skills = EXCLUDED.complementary_skills,
                                manifest_hash = EXCLUDED.manifest_hash,
                                domain = EXCLUDED.domain,
                                effort = EXCLUDED.effort,
                                descriptor = EXCLUDED.descriptor,
                                keywords = EXCLUDED.keywords,
                                updated_at = datetime('now')
                            """,
                            (
                                tool_name,
                                short_summary[:200],
                                full_profile,
                                json.dumps([]),
                                json.dumps([]),
                                manifest_hash,
                                domain,
                                effort,
                                descriptor,
                                keywords,
                            )
                        )

                        if embedding is not None:
                            row = cursor.execute(
                                "SELECT rowid FROM tool_capability_profiles WHERE tool_name = ?",
                                (tool_name,)
                            ).fetchone()
                            if row:
                                blob = _pack_embedding(embedding)
                                cursor.execute(
                                    "DELETE FROM tool_capability_profiles_vec WHERE rowid = ?",
                                    (row[0],)
                                )
                                cursor.execute(
                                    "INSERT INTO tool_capability_profiles_vec(rowid, embedding) VALUES (?, ?)",
                                    (row[0], blob)
                                )

                        cursor.close()
                    seeded += 1
                    logger.info(f"{LOG_PREFIX} seed_builtin_profiles: seeded {tool_name}")
                except Exception as e:
                    logger.error(f"{LOG_PREFIX} seed_builtin_profiles: DB upsert failed for {tool_name}: {e}")
                finally:
                    if db and not self._db:
                        db.close_pool()

            except Exception as e:
                logger.warning(f"{LOG_PREFIX} seed_builtin_profiles: failed for {tool_name}: {e}")

        if seeded:
            self._invalidate_cache()
            logger.info(f"{LOG_PREFIX} seed_builtin_profiles: seeded {seeded} profile(s)")

    def bootstrap_all(self) -> None:
        """
        Called on startup. Build profiles for any tool/skill that lacks one.
        Uses documentation field (or description fallback) + LLM profile builder.
        """
        logger.info(f"{LOG_PREFIX} Bootstrap: checking all tool/skill profiles...")
        self.seed_builtin_profiles()

        # Innate skills no longer profiled — documentation lives in TOOL_SCHEMA
        # dicts on each handler module. Procedural memory (strategy hints) is
        # separate and unaffected.

        # Bootstrap registered tools
        active_tool_names: set = set()
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            active_tool_names = set(registry.tools.keys())
            for tool_name, tool_data in registry.tools.items():
                try:
                    manifest = tool_data['manifest']
                    current_hash = _compute_manifest_hash(manifest)
                    if self.check_staleness(tool_name, current_hash):
                        self.build_profile(tool_name, manifest)
                    else:
                        # Rebuild if new columns were added but not yet populated
                        profile = self.get_full_profile(tool_name)
                        if profile and self._profile_needs_rebuild(profile):
                            logger.info(f"{LOG_PREFIX} Rebuilding tool {tool_name} profile (missing fields)")
                            self.build_profile(tool_name, manifest, force=True)
                        else:
                            logger.debug(f"{LOG_PREFIX} Tool {tool_name} profile is current")
                except Exception as e:
                    logger.warning(f"{LOG_PREFIX} Bootstrap failed for tool {tool_name}: {e}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Tool registry not available during bootstrap: {e}")

        # Purge profiles for tools/skills that no longer exist.
        # Preserve interface tool profiles — they're managed by the
        # interface lifecycle (register → profile, remove → unprofile).
        try:
            valid_names = ALL_SKILL_NAMES | active_tool_names

            # Interface tools may not be in registry.tools yet at bootstrap
            # time (daemons haven't re-registered). Read from DB directly.
            db = self._get_db()
            try:
                iface_rows = db.fetch_all("SELECT DISTINCT tool_name FROM interface_tools")
                valid_names |= {r['tool_name'] for r in (iface_rows or [])}
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} interface_tools table not available (expected on fresh install): {e}", exc_info=False)

            existing = db.fetch_all("SELECT tool_name FROM tool_capability_profiles")
            stale = [r['tool_name'] for r in (existing or []) if r['tool_name'] not in valid_names]
            if stale:
                placeholders = ','.join('?' * len(stale))
                db.execute(f"DELETE FROM tool_capability_profiles WHERE tool_name IN ({placeholders})", stale)
                logger.info(f"{LOG_PREFIX} Purged {len(stale)} stale profile(s): {stale}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to purge stale profiles: {e}")

        # Inject dynamic TOC into find_tools query description
        try:
            db = self._get_db()
            try:
                rows = db.fetch_all(
                    "SELECT keywords FROM tool_capability_profiles "
                    "WHERE tool_type = 'tool' AND keywords != '' ORDER BY tool_name"
                )
            finally:
                if not self._db:
                    db.close_pool()

            toc_parts = []
            for r in (rows or []):
                kw = r['keywords'] if isinstance(r, dict) else r[0]
                if kw:
                    first = kw.split(',')[0].strip()
                    if first and first not in toc_parts:
                        toc_parts.append(first)

            if toc_parts:
                toc_str = ",".join(sorted(toc_parts))
                from services.innate_skills.find_tools_skill import TOOL_SCHEMA as find_tools_schema
                find_tools_schema["input_schema"]["properties"]["query"]["description"] = (
                    f"Describe what you need or pick from: {toc_str}"
                )
                logger.info(f"{LOG_PREFIX} Injected TOC into find_tools: {toc_str}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} TOC injection failed (non-fatal): {e}")

        logger.info(f"{LOG_PREFIX} Bootstrap complete")

    # -- Helpers ---------------------------------------------------------------

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template from backend/prompts/."""
        import os
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        path = os.path.join(prompts_dir, f'{name}.md')
        with open(path, 'r') as f:
            return f.read()

    def _get_related_episodes(self, description: str, top_k: int = 20) -> str:
        """Query top K episodes semantically related to tool description."""
        try:
            emb_service = self._get_embedding_service()
            embedding = emb_service.generate_embedding(description)
            blob = _pack_embedding(embedding)
            if blob is None:
                return "No past interactions available."

            db = self._get_db()
            try:
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT e.outcome, e.gist, v.distance
                        FROM episodes_vec v
                        JOIN episodes e ON e.rowid = v.rowid
                        WHERE v.embedding MATCH ? AND k = ?
                          AND e.deleted_at IS NULL
                        ORDER BY v.distance
                        """,
                        (blob, top_k)
                    )
                    rows = cursor.fetchall()
                    cursor.close()

                if not rows:
                    return "No past interactions available."
                texts = []
                for r in rows:
                    text = (r['gist'] if isinstance(r, dict) else r[1]) or (r['outcome'] if isinstance(r, dict) else r[0]) or ''
                    if text:
                        texts.append(f"- {text[:200]}")
                return "\n".join(texts) if texts else "No past interactions available."
            finally:
                if not self._db:
                    db.close_pool()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Episode retrieval failed: {e}")
            return "No past interactions available."

    def _fallback_profile(self, tool_name: str, manifest: dict) -> dict:
        """Simple fallback profile when LLM is unavailable.

        Sets domain from the manifest's category field.
        """
        desc = manifest.get('documentation') or manifest.get('description', tool_name)
        domain = (manifest.get('category') or 'Other').replace('_', ' ').title()
        return {
            'short_summary': desc[:100],
            'full_profile': desc,
            'anti_scenarios': [],
            'complementary_skills': [],
            'domain': domain,
        }

    def _manifest_fallback_summaries(self) -> str:
        """Build triage summaries from tool manifests when DB has no profile rows.

        This is the safety net: if LLM-powered profile bootstrap hasn't run yet
        (or failed entirely), triage still learns about installed tools from
        their manifest declarations. Generic -- works for all tools.
        """
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            on_demand = registry.get_on_demand_tools()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Manifest fallback: registry unavailable: {e}")
            return ""

        if not on_demand:
            return ""

        by_domain = defaultdict(list)
        for tool_name in on_demand:
            tool_data = registry.tools.get(tool_name)
            if not tool_data:
                continue
            manifest = tool_data['manifest']
            fallback = self._fallback_profile(tool_name, manifest)
            domain = fallback.get('domain', 'Other')
            summary = fallback['short_summary'] + " (effort: moderate)"
            by_domain[domain].append(f"- {tool_name}: {summary}")

        if not by_domain:
            return ""

        lines = []
        for domain in sorted(by_domain.keys()):
            lines.append(f"## {domain}")
            lines.extend(by_domain[domain])
            lines.append("")

        result = "\n".join(lines).strip()
        logger.info(f"{LOG_PREFIX} Manifest fallback produced summaries for {sum(len(v) for v in by_domain.values())} tool(s)")
        return result

    @staticmethod
    def _profile_needs_rebuild(profile: dict) -> bool:
        """Check if a profile is missing fields added after initial build."""
        domain = profile.get('domain')
        if not domain or (domain == 'Other' and profile.get('tool_type') == 'tool'):
            return True
        if not profile.get('descriptor'):
            return True
        if not profile.get('keywords'):
            return True
        return False

    def _invalidate_cache(self):
        """Invalidate the triage summaries MemoryStore cache."""
        try:
            ms = self._get_store()
            ms.delete(TRIAGE_SUMMARIES_CACHE_KEY)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Cache invalidation failed (non-fatal): {e}", exc_info=False)
