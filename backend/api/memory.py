"""
Memory blueprint — /memory/context, /memory/forget, /memory/search.
"""

import logging
from flask import Blueprint, request, jsonify

from .auth import require_session

logger = logging.getLogger(__name__)

memory_bp = Blueprint('memory', __name__)


@memory_bp.route('/memory/context', methods=['GET'])
@require_session
def memory_context():
    """Return persistent life context — traits, facts, episodes, concepts."""
    try:
        from services.database_service import get_shared_db_service
        from services.user_trait_service import UserTraitService
        from services.fact_store_service import FactStoreService
        from services.episodic_retrieval_service import EpisodicRetrievalService
        from services.semantic_retrieval_service import SemanticRetrievalService
        from services.thread_service import get_thread_service
        from services.config_service import ConfigService

        db = get_shared_db_service()
        result = {"traits": [], "facts": [], "significant_episodes": [], "concepts": []}

        # User traits
        try:
            trait_service = UserTraitService(db)
            traits_text = trait_service.get_traits_for_prompt(user_id="primary")
            if traits_text:
                result["traits_summary"] = traits_text
        except Exception as e:
            logger.warning(f"[Memory] Trait retrieval failed: {e}")

        # Facts for active topic
        try:
            ts = get_thread_service()
            thread_id = ts.get_active_thread_id("default", "default")
            if thread_id:
                from services.redis_client import RedisClientService
                redis = RedisClientService.create_connection()
                topic_data = redis.hgetall(f"thread:{thread_id}")
                topic = topic_data.get("current_topic", "") if topic_data else ""
                if topic:
                    fact_service = FactStoreService()
                    facts = fact_service.get_all_facts(topic)
                    result["facts"] = [
                        {"key": f.get("key", ""), "value": f.get("value", ""), "confidence": f.get("confidence", 0)}
                        for f in facts
                    ]
        except Exception as e:
            logger.warning(f"[Memory] Fact retrieval failed: {e}")

        # High-salience episodes
        try:
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            retrieval = EpisodicRetrievalService(db, episodic_config)
            episodes = retrieval.retrieve_episodes("important events", limit=5)
            result["significant_episodes"] = [
                {"gist": ep.get("gist", ""), "salience": ep.get("salience", 0), "created_at": str(ep.get("created_at", ""))}
                for ep in episodes
            ]
        except Exception as e:
            logger.warning(f"[Memory] Episode retrieval failed: {e}")

        # Relevant concepts
        try:
            from services.embedding_service import EmbeddingService
            semantic = SemanticRetrievalService(db)
            concepts = semantic.retrieve_concepts("personal goals routines preferences", limit=5)
            result["concepts"] = [
                {"name": c.get("name", ""), "definition": c.get("definition", ""), "strength": c.get("strength", 0)}
                for c in concepts
            ]
        except Exception as e:
            logger.warning(f"[Memory] Concept retrieval failed: {e}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] memory/context error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve memory context"}), 500


@memory_bp.route('/memory/forget', methods=['POST'])
@require_session
def memory_forget():
    """Forget specific content by scope (topic, fact, all)."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    scope = data.get("scope", "")

    if scope not in ("topic", "fact", "all"):
        return jsonify({"error": "scope must be 'topic', 'fact', or 'all'"}), 400

    try:
        from services.gist_storage_service import GistStorageService
        from services.fact_store_service import FactStoreService
        from services.working_memory_service import WorkingMemoryService

        if scope == "topic":
            topic = data.get("topic", "").strip()
            if not topic:
                return jsonify({"error": "Missing 'topic' field"}), 400

            GistStorageService().clear_gists(topic)
            FactStoreService().clear_facts(topic)
            WorkingMemoryService().clear(topic)

            return jsonify({"deleted": True, "scope": "topic", "topic": topic}), 200

        elif scope == "fact":
            topic = data.get("topic", "").strip()
            fact_key = data.get("fact_key", "").strip()
            if not topic or not fact_key:
                return jsonify({"error": "Missing 'topic' or 'fact_key' field"}), 400

            fs = FactStoreService()
            # Delete specific fact key + remove from index
            redis_key = fs._get_fact_key(topic, fact_key)
            index_key = fs._get_fact_index_key(topic)
            fs.redis.delete(redis_key)
            fs.redis.zrem(index_key, fact_key)

            return jsonify({"deleted": True, "scope": "fact", "topic": topic, "fact_key": fact_key}), 200

        elif scope == "all":
            confirm = request.headers.get("X-Confirm-Delete", "")
            if confirm != "yes":
                return jsonify({"error": "Requires X-Confirm-Delete: yes header"}), 400

            # Clear all Redis memory stores
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()
            for pattern in ["working_memory:*", "gist:*", "gist_index:*", "fact:*", "fact_index:*", "world_state:*"]:
                keys = redis.keys(pattern)
                if keys:
                    redis.delete(*keys)

            # Truncate PostgreSQL tables
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                for table in ["episodes", "semantic_concepts", "semantic_relationships", "user_traits"]:
                    try:
                        conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                    except Exception:
                        pass
                conn.commit()

            # Audit trail
            try:
                from services.interaction_log_service import InteractionLogService
                from datetime import datetime, timezone
                log_service = InteractionLogService()
                log_service.log_event(
                    event_type="privacy_delete_all",
                    payload={"timestamp": datetime.now(timezone.utc).isoformat()}
                )
            except Exception:
                pass

            return jsonify({"deleted": True, "scope": "all"}), 200

    except Exception as e:
        logger.error(f"[REST API] memory/forget error: {e}", exc_info=True)
        return jsonify({"error": "Failed to forget"}), 500


@memory_bp.route('/memory/search', methods=['GET'])
@require_session
def memory_search():
    """Semantic search across all memory layers."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400

    try:
        from services.database_service import get_shared_db_service
        from services.episodic_retrieval_service import EpisodicRetrievalService
        from services.semantic_retrieval_service import SemanticRetrievalService
        from services.config_service import ConfigService

        db = get_shared_db_service()
        results = []

        # Episodic search
        try:
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            retrieval = EpisodicRetrievalService(db, episodic_config)
            episodes = retrieval.retrieve_episodes(query_text=query, limit=5)
            for ep in episodes:
                results.append({
                    "type": "episode",
                    "content": ep.get("gist", ""),
                    "score": ep.get("composite_score", ep.get("score", 0)),
                    "created_at": str(ep.get("created_at", "")),
                })
        except Exception as e:
            logger.warning(f"[Memory] Episode search failed: {e}")

        # Semantic concept search
        try:
            semantic = SemanticRetrievalService(db)
            concepts = semantic.retrieve_concepts(query=query, limit=5)
            for c in concepts:
                results.append({
                    "type": "concept",
                    "content": c.get("name", "") + ": " + c.get("definition", ""),
                    "score": c.get("score", c.get("similarity", 0)),
                    "strength": c.get("strength", 0),
                })
        except Exception as e:
            logger.warning(f"[Memory] Concept search failed: {e}")

        # Sort by score descending
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"[REST API] memory/search error: {e}", exc_info=True)
        return jsonify({"error": "Failed to search memory"}), 500
