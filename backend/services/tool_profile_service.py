"""
Tool Profile Service — Builds, stores, queries, and enriches tool capability profiles.

Profiles are LLM-generated structured descriptions of what each tool/skill does,
when to use it, and example usage scenarios. Used by CognitiveTriageService to
inject rich capability context into the triage LLM prompt.

Profiles are stored in tool_capability_profiles PostgreSQL table with:
- short_summary: one-sentence description for triage prompt injection
- full_profile: detailed description for ACT prompt injection
- usage_scenarios: up to 50 scenarios for semantic matching
- embedding: 768-dim vector for cosine similarity

Bootstrap: called on startup to build profiles for any missing tool/skill.
Enrichment: triggered by high-salience episodes or idle-time background service.
"""

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TOOL PROFILE]"

# Innate skill descriptions
SKILL_DESCRIPTIONS = {
    'recall': 'Search memory, retrieve stored information, look up what Chalie knows about a topic or person',
    'memorize': 'Store information, save a note, remember a fact, keep something for later',
    'introspect': 'Self-examine internal state, check how much is known about a topic, inspect confidence',
    'associate': 'Find related concepts, explore connections, brainstorm associations between ideas',
    'list': 'Manage named lists: add, remove, check off, or view items in shopping, to-do, and other lists',
    'schedule': 'Set reminders, schedule tasks, create appointments and recurring events',
    'focus': 'Start and manage deep focus or work sessions, Pomodoro-style timers',
    'autobiography': 'Generate a personal autobiography or life summary based on stored memories',
}

# Redis cache key and TTL
TRIAGE_SUMMARIES_CACHE_KEY = "tool_triage_summaries"
TRIAGE_SUMMARIES_TTL = 300  # 5 minutes

MAX_SCENARIOS = 50
MIN_SCENARIO_DISTANCE = 0.12  # cosine distance for deduplication


def _compute_manifest_hash(manifest: dict) -> str:
    """MD5 hash of manifest for staleness detection."""
    content = json.dumps(manifest, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


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
        from services.database_service import get_lightweight_db_service
        return get_lightweight_db_service()

    def _get_llm(self):
        from services.llm_service import create_llm_service
        from services.config_service import ConfigService
        agent_cfg = ConfigService.resolve_agent_config('cognitive-triage')
        return create_llm_service(agent_cfg)

    def _get_embedding_service(self):
        from services.embedding_service import EmbeddingService
        return EmbeddingService()

    def _get_redis(self):
        from services.redis_client import RedisClientService
        return RedisClientService.create_connection(decode_responses=True)

    # ── Profile Building ──────────────────────────────────────────────

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

        # Build LLM prompt
        prompt_template = self._load_prompt('tool-profile-builder')
        prompt = (
            prompt_template
            .replace('{{manifest}}', json.dumps(manifest, indent=2))
            .replace('{{episodes}}', episodes_text)
        )

        try:
            llm = self._get_llm()
            response_text = llm.send_message("", prompt).text
            profile_data = _extract_json(response_text)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM profile build failed for {tool_name}: {e}")
            profile_data = self._fallback_profile(tool_name, manifest)

        # Cap scenarios
        usage_scenarios = profile_data.get('usage_scenarios', [])[:MAX_SCENARIOS]
        anti_scenarios = profile_data.get('anti_scenarios', [])[:20]

        # Generate embedding from full_profile
        embedding = None
        try:
            emb_service = self._get_embedding_service()
            full_profile = profile_data.get('full_profile', description)
            embedding = emb_service.generate_embedding(full_profile)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding generation failed for {tool_name}: {e}")

        # Upsert into database
        db = self._get_db()
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
            triage_triggers = profile_data.get('triage_triggers', [])[:10]
            db.execute(
                """
                INSERT INTO tool_capability_profiles
                    (tool_name, tool_type, short_summary, full_profile, usage_scenarios,
                     anti_scenarios, complementary_skills, embedding, manifest_hash, domain,
                     triage_triggers, updated_at)
                VALUES (%s, 'tool', %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (tool_name) DO UPDATE SET
                    tool_type = 'tool',
                    short_summary = EXCLUDED.short_summary,
                    full_profile = EXCLUDED.full_profile,
                    usage_scenarios = EXCLUDED.usage_scenarios,
                    anti_scenarios = EXCLUDED.anti_scenarios,
                    complementary_skills = EXCLUDED.complementary_skills,
                    embedding = EXCLUDED.embedding,
                    manifest_hash = EXCLUDED.manifest_hash,
                    domain = EXCLUDED.domain,
                    triage_triggers = EXCLUDED.triage_triggers,
                    updated_at = NOW()
                """,
                (
                    tool_name,
                    profile_data.get('short_summary', f'{tool_name} tool')[:100],
                    profile_data.get('full_profile', description),
                    json.dumps(usage_scenarios),
                    json.dumps(anti_scenarios),
                    json.dumps(profile_data.get('complementary_skills', [])),
                    embedding_str,
                    manifest_hash,
                    profile_data.get('domain', 'Other'),
                    json.dumps(triage_triggers),
                )
            )
            logger.info(f"{LOG_PREFIX} Upserted profile for {tool_name}")
        except Exception as e:
            logger.error(f"{LOG_PREFIX} DB upsert failed for {tool_name}: {e}")
        finally:
            if not self._db:
                db.close_pool()

        # Invalidate triage summaries cache
        self._invalidate_cache()

        return profile_data

    def build_skill_profile(self, skill_name: str, skill_desc: str, force: bool = False) -> dict:
        """Build and store a capability profile for an innate skill."""
        logger.info(f"{LOG_PREFIX} Building profile for skill: {skill_name}")

        # Use skill description as a minimal manifest
        pseudo_manifest = {
            'name': skill_name,
            'description': skill_desc,
            'documentation': skill_desc,
        }
        manifest_hash = _compute_manifest_hash(pseudo_manifest)

        if not force and not self.check_staleness(skill_name, manifest_hash):
            logger.info(f"{LOG_PREFIX} Profile for skill {skill_name} is current, skipping")
            return self.get_full_profile(skill_name) or {}

        episodes_text = self._get_related_episodes(skill_desc)

        prompt_template = self._load_prompt('tool-profile-builder')
        prompt = (
            prompt_template
            .replace('{{manifest}}', json.dumps(pseudo_manifest, indent=2))
            .replace('{{episodes}}', episodes_text)
        )

        try:
            llm = self._get_llm()
            response_text = llm.send_message("", prompt).text
            profile_data = _extract_json(response_text)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM profile build failed for skill {skill_name}: {e}")
            profile_data = {
                'short_summary': skill_desc[:100],
                'full_profile': skill_desc,
                'usage_scenarios': [],
                'anti_scenarios': [],
                'complementary_skills': [],
            }

        usage_scenarios = profile_data.get('usage_scenarios', [])[:MAX_SCENARIOS]

        embedding = None
        try:
            emb_service = self._get_embedding_service()
            embedding = emb_service.generate_embedding(profile_data.get('full_profile', skill_desc))
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding failed for skill {skill_name}: {e}")

        db = self._get_db()
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
            triage_triggers = profile_data.get('triage_triggers', [])[:10]
            db.execute(
                """
                INSERT INTO tool_capability_profiles
                    (tool_name, tool_type, short_summary, full_profile, usage_scenarios,
                     anti_scenarios, complementary_skills, embedding, manifest_hash, domain,
                     triage_triggers, updated_at)
                VALUES (%s, 'skill', %s, %s, %s, %s, %s, %s, %s, 'Innate Skill', %s, NOW())
                ON CONFLICT (tool_name) DO UPDATE SET
                    tool_type = 'skill',
                    short_summary = EXCLUDED.short_summary,
                    full_profile = EXCLUDED.full_profile,
                    usage_scenarios = EXCLUDED.usage_scenarios,
                    anti_scenarios = EXCLUDED.anti_scenarios,
                    complementary_skills = EXCLUDED.complementary_skills,
                    embedding = EXCLUDED.embedding,
                    manifest_hash = EXCLUDED.manifest_hash,
                    domain = EXCLUDED.domain,
                    triage_triggers = EXCLUDED.triage_triggers,
                    updated_at = NOW()
                """,
                (
                    skill_name,
                    profile_data.get('short_summary', skill_desc[:100]),
                    profile_data.get('full_profile', skill_desc),
                    json.dumps(usage_scenarios),
                    json.dumps(profile_data.get('anti_scenarios', [])[:20]),
                    json.dumps(profile_data.get('complementary_skills', [])),
                    embedding_str,
                    manifest_hash,
                    json.dumps(triage_triggers),
                )
            )
            logger.info(f"{LOG_PREFIX} Upserted profile for skill {skill_name}")
        except Exception as e:
            logger.error(f"{LOG_PREFIX} DB upsert failed for skill {skill_name}: {e}")
        finally:
            if not self._db:
                db.close_pool()

        self._invalidate_cache()
        return profile_data

    # ── Enrichment ────────────────────────────────────────────────────

    def enrich_from_episodes(self, tool_name: str, episode_ids: list) -> int:
        """Enrich tool profile with new scenarios from episodes. Returns count of new scenarios added."""
        profile = self.get_full_profile(tool_name)
        if not profile:
            logger.warning(f"{LOG_PREFIX} No profile found for {tool_name}, cannot enrich")
            return 0

        existing_scenarios = profile.get('usage_scenarios', [])

        # Fetch episode content
        episodes_text = self._get_episodes_by_ids(episode_ids)
        if not episodes_text:
            return 0

        prompt_template = self._load_prompt('tool-enrichment')
        prompt = (
            prompt_template
            .replace('{{tool_name}}', tool_name)
            .replace('{{full_profile}}', profile.get('full_profile', ''))
            .replace('{{existing_scenarios}}', json.dumps(existing_scenarios[:20], indent=2))
            .replace('{{episodes}}', episodes_text)
        )

        try:
            llm = self._get_llm()
            response_text = llm.send_message("", prompt).text
            result = json.loads(response_text)
            new_scenarios = result.get('new_scenarios', [])
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Enrichment LLM call failed for {tool_name}: {e}")
            return 0

        if not new_scenarios:
            return 0

        # Quality filter: semantic distinctness
        accepted = self._filter_distinct_scenarios(new_scenarios, existing_scenarios)
        if not accepted:
            return 0

        # Merge and cap
        merged = existing_scenarios + accepted
        if len(merged) > MAX_SCENARIOS:
            merged = merged[:MAX_SCENARIOS]

        # Refresh embedding
        embedding = None
        try:
            emb_service = self._get_embedding_service()
            profile_text = profile.get('full_profile', '') + ' ' + ' '.join(merged[:10])
            embedding = emb_service.generate_embedding(profile_text)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding refresh failed for {tool_name}: {e}")

        db = self._get_db()
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]" if embedding else None
            current_ids = profile.get('enrichment_episode_ids', [])
            new_ids = list(set(current_ids + episode_ids))

            update_sql = """
                UPDATE tool_capability_profiles
                SET usage_scenarios = %s,
                    enrichment_episode_ids = %s,
                    enrichment_count = enrichment_count + 1,
                    last_enriched_at = NOW(),
                    updated_at = NOW()
                    {embedding_clause}
                WHERE tool_name = %s
            """
            if embedding_str:
                update_sql = update_sql.replace('{embedding_clause}', ', embedding = %s')
                db.execute(update_sql, (json.dumps(merged), json.dumps(new_ids), embedding_str, tool_name))
            else:
                update_sql = update_sql.replace('{embedding_clause}', '')
                db.execute(update_sql, (json.dumps(merged), json.dumps(new_ids), tool_name))

            logger.info(f"{LOG_PREFIX} Enriched {tool_name} with {len(accepted)} new scenarios")
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Enrichment DB update failed for {tool_name}: {e}")
        finally:
            if not self._db:
                db.close_pool()

        self._invalidate_cache()
        return len(accepted)

    def check_episode_relevance(self, episode_embedding, episode_id: str) -> None:
        """Check if an episode is relevant to any tool profile; enrich if so."""
        db = self._get_db()
        try:
            embedding_str = f"[{','.join(str(x) for x in episode_embedding)}]"
            rows = db.fetch_all(
                """
                SELECT tool_name, 1 - (embedding <=> %s::vector) AS similarity
                FROM tool_capability_profiles
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT 1
                """,
                (embedding_str,)
            )
            if rows and rows[0]['similarity'] > 0.7:
                tool_name = rows[0]['tool_name']
                logger.info(
                    f"{LOG_PREFIX} Episode {episode_id} relevant to {tool_name} "
                    f"(similarity={rows[0]['similarity']:.3f}), triggering enrichment"
                )
                self.enrich_from_episodes(tool_name, [episode_id])
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Episode relevance check failed: {e}")
        finally:
            if not self._db:
                db.close_pool()

    # ── Query ─────────────────────────────────────────────────────────

    def get_triage_summaries(self) -> str:
        """
        Get pre-formatted tool summaries grouped by domain for triage prompt injection.
        Cached in Redis for 5 minutes.
        """
        try:
            redis = self._get_redis()
            cached = redis.get(TRIAGE_SUMMARIES_CACHE_KEY)
            if cached:
                return cached
        except Exception:
            pass

        summaries = self._build_triage_summaries()

        try:
            redis = self._get_redis()
            redis.setex(TRIAGE_SUMMARIES_CACHE_KEY, TRIAGE_SUMMARIES_TTL, summaries)
        except Exception:
            pass

        return summaries

    def _build_triage_summaries(self) -> str:
        """Build domain-grouped tool summaries from profiles table (DB-driven, tool-agnostic)."""
        db = self._get_db()
        try:
            rows = db.fetch_all(
                "SELECT tool_name, tool_type, short_summary, triage_triggers, domain "
                "FROM tool_capability_profiles ORDER BY domain, tool_name"
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to fetch profiles: {e}")
            return ""
        finally:
            if not self._db:
                db.close_pool()

        if not rows:
            return ""

        # Filter to only external tools (skills are always available, not listed in triage)
        by_domain = defaultdict(list)
        for r in rows:
            if r['tool_type'] == 'tool':
                domain = r.get('domain') or 'Other'
                summary = r['short_summary']
                triggers = r.get('triage_triggers') or []
                if isinstance(triggers, str):
                    triggers = json.loads(triggers)
                if triggers:
                    summary += f" [{', '.join(triggers[:10])}]"
                by_domain[domain].append(f"- {r['tool_name']}: {summary}")

        if not by_domain:
            return ""

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
                "SELECT * FROM tool_capability_profiles WHERE tool_name = %s",
                (tool_name,)
            )
            if rows:
                row = dict(rows[0])
                # Parse JSONB fields
                for field in ('usage_scenarios', 'anti_scenarios', 'complementary_skills', 'enrichment_episode_ids', 'triage_triggers'):
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
            placeholders = ','.join(['%s'] * len(tool_names))
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
                "SELECT manifest_hash FROM tool_capability_profiles WHERE tool_name = %s",
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

    def bootstrap_all(self) -> None:
        """
        Called on startup. Build profiles for any tool/skill that lacks one.
        Uses documentation field (or description fallback) + LLM profile builder.
        """
        logger.info(f"{LOG_PREFIX} Bootstrap: checking all tool/skill profiles...")

        # Bootstrap innate skills
        for skill_name, skill_desc in SKILL_DESCRIPTIONS.items():
            try:
                pseudo_manifest = {'name': skill_name, 'documentation': skill_desc}
                manifest_hash = _compute_manifest_hash(pseudo_manifest)
                if self.check_staleness(skill_name, manifest_hash):
                    self.build_skill_profile(skill_name, skill_desc)
                else:
                    # Rebuild if new columns were added but not yet populated
                    profile = self.get_full_profile(skill_name)
                    if profile and self._profile_needs_rebuild(profile):
                        logger.info(f"{LOG_PREFIX} Rebuilding skill {skill_name} profile (missing fields)")
                        self.build_skill_profile(skill_name, skill_desc, force=True)
                    else:
                        logger.debug(f"{LOG_PREFIX} Skill {skill_name} profile is current")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Bootstrap failed for skill {skill_name}: {e}")

        # Bootstrap registered tools
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
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

        logger.info(f"{LOG_PREFIX} Bootstrap complete")

    # ── Helpers ───────────────────────────────────────────────────────

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
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"

            db = self._get_db()
            try:
                rows = db.fetch_all(
                    """
                    SELECT outcome, gist, 1 - (embedding <=> %s::vector) AS similarity
                    FROM episodes
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (embedding_str, top_k)
                )
                if not rows:
                    return "No past interactions available."
                texts = []
                for r in rows:
                    text = r.get('gist') or r.get('outcome', '')
                    if text:
                        texts.append(f"- {text[:200]}")
                return "\n".join(texts) if texts else "No past interactions available."
            finally:
                if not self._db:
                    db.close_pool()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Episode retrieval failed: {e}")
            return "No past interactions available."

    def _get_episodes_by_ids(self, episode_ids: list) -> str:
        """Fetch episode content by IDs."""
        if not episode_ids:
            return ""
        db = self._get_db()
        try:
            placeholders = ','.join(['%s'] * len(episode_ids))
            rows = db.fetch_all(
                f"SELECT outcome, gist FROM episodes WHERE id::text IN ({placeholders})",
                tuple(str(eid) for eid in episode_ids)
            )
            if not rows:
                return ""
            texts = []
            for r in rows:
                text = r.get('gist') or r.get('outcome', '')
                if text:
                    texts.append(f"- {text[:200]}")
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Episode fetch by IDs failed: {e}")
            return ""
        finally:
            if not self._db:
                db.close_pool()

    def _filter_distinct_scenarios(self, new_scenarios: list, existing_scenarios: list) -> list:
        """Filter new scenarios to only those semantically distinct from existing ones."""
        if not existing_scenarios:
            return new_scenarios

        try:
            emb_service = self._get_embedding_service()
            existing_vecs = [emb_service.generate_embedding(s) for s in existing_scenarios[:20]]

            import numpy as np
            accepted = []
            for scenario in new_scenarios:
                vec = emb_service.generate_embedding(scenario)
                vec_np = np.array(vec)
                norm = np.linalg.norm(vec_np)
                if norm > 0:
                    vec_np = vec_np / norm

                is_distinct = True
                for existing_vec in existing_vecs:
                    ex_np = np.array(existing_vec)
                    ex_norm = np.linalg.norm(ex_np)
                    if ex_norm > 0:
                        ex_np = ex_np / ex_norm
                    similarity = float(np.dot(vec_np, ex_np))
                    distance = 1.0 - similarity
                    if distance < MIN_SCENARIO_DISTANCE:
                        is_distinct = False
                        break

                if is_distinct:
                    accepted.append(scenario)
                    existing_vecs.append(vec)  # Add to existing to avoid near-duplicates within batch

            return accepted
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Scenario filtering failed: {e}")
            return new_scenarios  # Accept all if filtering fails

    def _fallback_profile(self, tool_name: str, manifest: dict) -> dict:
        """Simple fallback profile when LLM is unavailable."""
        desc = manifest.get('documentation') or manifest.get('description', tool_name)
        return {
            'short_summary': desc[:100],
            'full_profile': desc,
            'usage_scenarios': [ex.get('description', '') for ex in manifest.get('examples', [])[:10] if ex.get('description')],
            'anti_scenarios': [],
            'complementary_skills': [],
            'triage_triggers': [],
        }

    @staticmethod
    def _profile_needs_rebuild(profile: dict) -> bool:
        """Check if a profile is missing fields added after initial build."""
        domain = profile.get('domain')
        if not domain or (domain == 'Other' and profile.get('tool_type') == 'tool'):
            return True
        triggers = profile.get('triage_triggers')
        if not triggers or triggers == []:
            return True
        scenarios = profile.get('usage_scenarios')
        if not scenarios or scenarios == []:
            return True
        return False

    def _invalidate_cache(self):
        """Invalidate the triage summaries Redis cache."""
        try:
            redis = self._get_redis()
            redis.delete(TRIAGE_SUMMARIES_CACHE_KEY)
        except Exception:
            pass
