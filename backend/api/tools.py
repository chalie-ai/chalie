"""
Tools blueprint — /tools endpoints for listing tools and managing their configuration.
"""

import json
import logging
from pathlib import Path
from urllib.parse import quote as url_quote

from flask import Blueprint, request, jsonify

from .auth import require_session

logger = logging.getLogger(__name__)

tools_bp = Blueprint("tools", __name__)


def _check_webhook_rate_limit(tool_name: str) -> bool:
    """Return True if within rate limit (30 req/min per tool), False if exceeded."""
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        key = f"webhook_rate:{tool_name}"
        count = store.incr(key)
        if count == 1:
            store.expire(key, 60)  # 1-minute sliding window
        return count <= 30
    except Exception:
        return True  # Fail open on MemoryStore errors


@tools_bp.route("/tools/webhook/<tool_name>", methods=["POST"])
def tool_webhook(tool_name):
    """
    Webhook endpoint for tool invocation. No session required.

    Auth: X-Chalie-Token header (simple key) or X-Chalie-Signature +
          X-Chalie-Timestamp (HMAC-SHA256 with replay protection).
    Rate limit: 30 req/min per tool.
    Payload size limit: 512KB.
    """
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        # Payload size guard (512KB)
        content_length = request.content_length
        if content_length and content_length > 512 * 1024:
            return jsonify({"error": "Payload too large (max 512KB)"}), 413

        raw_body = request.get_data(cache=True)
        if len(raw_body) > 512 * 1024:
            return jsonify({"error": "Payload too large (max 512KB)"}), 413

        # Tool existence + trigger type check
        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if not tool:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        trigger = tool["manifest"].get("trigger", {})
        if trigger.get("type") != "webhook":
            return jsonify({"error": f"Tool '{tool_name}' does not accept webhooks"}), 404

        # Auth: HMAC-SHA256 (preferred) or simple token
        db = get_shared_db_service()
        config_svc = ToolConfigService(db)

        signature = request.headers.get("X-Chalie-Signature", "")
        timestamp = request.headers.get("X-Chalie-Timestamp", "")
        token = request.headers.get("X-Chalie-Token", "")

        auth_ok = False
        if signature and timestamp:
            auth_ok = config_svc.validate_webhook_hmac(tool_name, timestamp, raw_body, signature)
        elif token:
            auth_ok = config_svc.validate_webhook_key(tool_name, token)

        if not auth_ok:
            logger.warning(f"[TOOLS API] Webhook auth failed for '{tool_name}'")
            return jsonify({"error": "Unauthorized"}), 403

        # Rate limiting
        if not _check_webhook_rate_limit(tool_name):
            return jsonify({"error": "Rate limit exceeded (30 req/min)"}), 429

        # Parse body
        try:
            webhook_body = request.get_json(force=True) or {}
        except Exception:
            webhook_body = {}

        # Dialog callback — routes "tool" output through full cognitive pipeline
        trigger_prompt = trigger.get("prompt", "")
        dialog_turns = []

        def _dialog_callback(result):
            from workers.digest_worker import process_tool_dialog
            request_text = result.get("text", "")
            response = process_tool_dialog(
                text=request_text,
                tool_name=tool_name,
                trigger_prompt=trigger_prompt,
            )
            dialog_turns.append({"request": request_text, "response": response})
            return response

        # Invoke the tool
        result = registry.invoke_webhook(tool_name, webhook_body, dialog_callback=_dialog_callback)

        # Store final-turn dialog memory if interactive turns occurred
        if dialog_turns:
            try:
                from workers.digest_worker import store_tool_dialog_memory
                store_tool_dialog_memory(tool_name, dialog_turns)
            except Exception as e:
                logger.warning(f"[TOOLS API] Dialog memory store failed for '{tool_name}': {e}")

        # Route output
        output_type = result.get("output") if isinstance(result, dict) else None

        if output_type == "prompt":
            result_text = result.get("text", "")
            if result_text:
                try:
                    from services.prompt_queue import PromptQueue
                    from workers.digest_worker import digest_worker
                    prompt_template = trigger_prompt or f"Tool {tool_name} says:"
                    full_prompt = f"{prompt_template}\n\n--- Tool Data ---\n{result_text[:3000]}"
                    queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
                    queue.enqueue(full_prompt, {
                        "source": f"webhook_tool:{tool_name}",
                        "tool_name": tool_name,
                        "destination": "web",
                        "priority": result.get("priority", "normal"),
                    })
                except Exception as e:
                    logger.warning(f"[TOOLS API] Webhook prompt enqueue failed for '{tool_name}': {e}")

        logger.info(f"[TOOLS API] Webhook '{tool_name}' processed (output={output_type!r})")
        return jsonify({"status": "ok"}), 200

    except Exception as e:
        logger.error(f"[TOOLS API] Webhook error for '{tool_name}': {e}", exc_info=True)
        return jsonify({"error": "Internal error"}), 500


@tools_bp.route("/tools/<tool_name>/webhook/key", methods=["POST"])
@require_session
def generate_webhook_key(tool_name: str):
    """
    Generate (or regenerate) a webhook API key for a tool.

    Returns the key once — caller must store it securely.
    Subsequent calls invalidate the previous key.
    """
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if not tool:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        trigger_type = tool["manifest"].get("trigger", {}).get("type")
        if trigger_type != "webhook":
            return jsonify({"error": f"Tool '{tool_name}' is not a webhook tool"}), 400

        db = get_shared_db_service()
        key = ToolConfigService(db).generate_webhook_key(tool_name)
        webhook_url = f"/api/tools/webhook/{tool_name}"

        return jsonify({
            "tool_name": tool_name,
            "webhook_key": key,
            "webhook_url": webhook_url,
        }), 200

    except Exception as e:
        logger.error(f"[TOOLS API] Webhook key generation error for '{tool_name}': {e}", exc_info=True)
        return jsonify({"error": "Failed to generate webhook key"}), 500


@tools_bp.route("/tools/catalog", methods=["GET"])
@require_session
def list_tools_catalog():
    """List all discovered tools with metadata, including non-installed ones.

    Unlike GET /tools (installed+ready only), this returns every tool Chalie
    has discovered from its tools directory, with an ``installed`` flag.

    Returns:
        200 with ``{"catalog": [{name, installed, category, description,
                                  documentation, icon, version, parameters}]}``
    """
    from services.tool_registry_service import ToolRegistryService
    registry = ToolRegistryService()
    catalog = []
    for name, tool_data in registry.tools.items():
        manifest = tool_data.get('manifest', {})
        entry = {
            'name': name,
            'installed': registry._is_ready(name, tool_data),
            'category': manifest.get('category', 'general'),
            'description': manifest.get('description', ''),
            'documentation': manifest.get('documentation', ''),
            'icon': manifest.get('icon', ''),
            'version': manifest.get('version', ''),
            'parameters': manifest.get('parameters', {}),
            'trigger': manifest.get('trigger', {}),
        }
        catalog.append(entry)
    catalog.sort(key=lambda x: x['name'])
    return jsonify({'catalog': catalog})


@tools_bp.route("/tools", methods=["GET"])
@require_session
def list_tools():
    """
    List all tools: loaded (connected/available/system), building, error, and disabled.

    Returns:
        {
            "tools": [
                {
                    "name": str,
                    "status": "connected|available|system|disabled|building|error",
                    "icon": str,
                    "description": str,
                    "last_error": str|null,
                    ...
                }
            ],
            "count": int
        }
    """
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        registry = ToolRegistryService()
        tools_dir = registry.tools_dir

        # DB access is best-effort
        try:
            db = get_shared_db_service()
            tool_config_svc = ToolConfigService(db)
        except Exception as db_err:
            logger.warning(f"[REST API] tools list: DB unavailable: {db_err}")
            tool_config_svc = None

        result = []
        processed_names = set()

        # 1. Active tools in registry (connected/available/system)
        for name in sorted(registry.tools.keys()):
            tool = registry.tools[name]
            tool_dir = Path(tool["dir"])

            # Ghost check: verify tool directory still exists on disk
            if not tool_dir.exists():
                logger.debug(f"[REST API] Skipping ghost tool '{name}': directory not found at {tool_dir}")
                continue

            manifest = tool["manifest"]
            trigger = manifest.get("trigger", {})

            display_name = name.replace("_", " ").title()
            icon = manifest.get("icon", "⚙")

            schema_dict = manifest.get("config_schema", {})
            # Handle array format by converting to dict
            if isinstance(schema_dict, list):
                schema_dict = {item.get("key"): item for item in schema_dict if isinstance(item, dict) and "key" in item}

            stored_config = tool_config_svc.get_tool_config(name) if tool_config_svc else {}

            has_secret_fields = any(v.get("secret", False) for v in schema_dict.values())
            uses_oauth = manifest.get("auth", {}).get("type") == "oauth2"
            if not has_secret_fields:
                status = "system"
            elif uses_oauth:
                # OAuth tools: "connected" only when tokens are present
                if stored_config.get("_oauth_access_token"):
                    status = "connected"
                elif stored_config:
                    status = "available"  # config saved but OAuth not completed
                else:
                    status = "available"
            elif stored_config:
                status = "connected"
            else:
                status = "available"

            tool_entry = {
                "name": name,
                "display_name": display_name,
                "icon": icon,
                "description": manifest.get("description", ""),
                "trigger_type": trigger.get("type", ""),
                "status": status,
                "config_keys": [k for k in stored_config.keys() if k not in ToolConfigService.RESERVED_KEYS],
                "has_sandbox": bool(manifest.get("sandbox")),
                "last_error": None,
                "source_type": stored_config.get("_source_type"),
                "source_url": stored_config.get("_source_url"),
                "installed_tag": stored_config.get("_installed_tag"),
            }
            if trigger.get("type") == "webhook":
                tool_entry["webhook_url"] = f"/api/tools/webhook/{name}"
                tool_entry["webhook_key_set"] = bool(stored_config.get("_webhook_key"))
            # OAuth status — generic, reads from manifest auth block
            auth_block = manifest.get("auth", {})
            if auth_block.get("type"):
                tool_entry["auth_type"] = auth_block["type"]
                tool_entry["auth_provider_hint"] = auth_block.get("provider_hint", "")
                tool_entry["oauth_connected"] = bool(stored_config.get("_oauth_access_token"))
            result.append(tool_entry)
            processed_names.add(name)

        # 2. Building/error tools (in registry but not yet loaded, or failed)
        build_statuses = registry.get_all_build_statuses()
        for name, status_info in build_statuses.items():
            if name in processed_names:
                continue

            status = status_info.get("status", "unknown")
            error = status_info.get("error")

            # Try to read manifest from tools/ directory for metadata
            tool_dir = tools_dir / name
            manifest = {}
            icon = "⚙"
            description = ""

            if tool_dir.exists():
                manifest_path = tool_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                        icon = manifest.get("icon", "⚙")
                        description = manifest.get("description", "")
                    except Exception:
                        pass

            # Source metadata (best-effort from DB)
            build_config = tool_config_svc.get_tool_config(name) if tool_config_svc else {}

            result.append({
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "icon": icon,
                "description": description,
                "trigger_type": manifest.get("trigger", {}).get("type", ""),
                "status": status,
                "last_error": error,
                "source_type": build_config.get("_source_type"),
                "source_url": build_config.get("_source_url"),
                "installed_tag": build_config.get("_installed_tag"),
            })
            processed_names.add(name)

        # 3. Filesystem scan: disabled tools + safety net for tools that failed to load
        if tools_dir.exists():
            for tool_dir in sorted(tools_dir.iterdir()):
                if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                    continue

                manifest_path = tool_dir / "manifest.json"
                # Trusted tools use runner.py; only sandboxed tools need a Dockerfile
                if not manifest_path.exists():
                    continue
                if not (tool_dir / "Dockerfile").exists() and not (tool_dir / "runner.py").exists():
                    continue

                try:
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    name = manifest.get("name", tool_dir.name)
                except Exception:
                    name = tool_dir.name
                    manifest = {}

                if name in processed_names:
                    continue

                stored_config = tool_config_svc.get_tool_config(name) if tool_config_svc else {}
                if stored_config.get("_enabled", "true").lower() == "false":
                    status = "disabled"
                    last_error = None
                else:
                    status = "error"
                    last_error = "Tool failed to load at startup"

                icon = manifest.get("icon", "⚙")
                description = manifest.get("description", "")

                result.append({
                    "name": name,
                    "display_name": name.replace("_", " ").title(),
                    "icon": icon,
                    "description": description,
                    "trigger_type": manifest.get("trigger", {}).get("type", ""),
                    "status": status,
                    "last_error": last_error,
                    "source_type": stored_config.get("_source_type"),
                    "source_url": stored_config.get("_source_url"),
                    "installed_tag": stored_config.get("_installed_tag"),
                })
                processed_names.add(name)

        # Sort result by name
        result.sort(key=lambda t: t["name"])

        return jsonify({"tools": result, "count": len(result)}), 200

    except Exception as e:
        logger.error(f"[REST API] tools list error: {e}", exc_info=True)
        return jsonify({"error": "Failed to list tools"}), 500


@tools_bp.route("/tools/<tool_name>/config", methods=["GET"])
@require_session
def get_tool_config(tool_name: str):
    """Get current config for a tool (secrets masked)."""
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        registry = ToolRegistryService()
        if tool_name not in registry.tools:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        schema = registry.get_tool_config_schema(tool_name)
        db = get_shared_db_service()
        config = ToolConfigService(db).get_tool_config(tool_name)

        # Mask secrets in response; filter internal reserved keys
        masked = {}
        for key, value in config.items():
            if key in ToolConfigService.RESERVED_KEYS:
                continue
            field_def = schema.get(key, {})
            masked[key] = "***" if field_def.get("secret", False) else value

        # Enrich schema with UI-friendly fields (label, hint, placeholder)
        # The raw manifest schema uses "description" and "default"; the brain UI
        # expects "label", "hint", and "placeholder" — add them here.
        enriched_schema = {}
        for key, field_def in schema.items():
            if isinstance(field_def, dict):
                enriched_schema[key] = {
                    **field_def,
                    "label": field_def.get("description", key),
                    "hint": field_def.get("description", ""),
                    "placeholder": field_def.get("default", ""),
                }
            else:
                enriched_schema[key] = field_def

        return jsonify({
            "tool_name": tool_name,
            "config_schema": enriched_schema,
            "config": masked,
        }), 200

    except Exception as e:
        logger.error(f"[REST API] tools config GET error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve tool config"}), 500


@tools_bp.route("/tools/<tool_name>/config", methods=["PUT"])
@require_session
def set_tool_config(tool_name: str):
    """Set config keys for a tool. Validates against config_schema."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        registry = ToolRegistryService()
        if tool_name not in registry.tools:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        schema = registry.get_tool_config_schema(tool_name)
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        # Reject unknown keys if schema is defined
        if schema:
            unknown = set(data.keys()) - set(schema.keys())
            if unknown:
                return jsonify({"error": f"Unknown config keys: {sorted(unknown)}"}), 400

        if not data:
            return jsonify({"error": "No config keys provided"}), 400

        db = get_shared_db_service()
        try:
            saved = ToolConfigService(db).set_tool_config(tool_name, data)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        if not saved:
            logger.error(f"[REST API] ToolConfigService.set_tool_config returned False for {tool_name}")
            return jsonify({"error": "Failed to save config"}), 500

        return jsonify({"saved": True, "tool_name": tool_name, "keys": sorted(data.keys())}), 200

    except Exception as e:
        logger.error(f"[REST API] tools config PUT error: {e}", exc_info=True)
        return jsonify({"error": f"Failed to set tool config: {str(e)}"}), 500


@tools_bp.route("/tools/<tool_name>/config/<key>", methods=["DELETE"])
@require_session
def delete_tool_config_key(tool_name: str, key: str):
    """Delete a single config key for a tool."""
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        registry = ToolRegistryService()
        if tool_name not in registry.tools:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        db = get_shared_db_service()
        deleted = ToolConfigService(db).delete_tool_config_key(tool_name, key)

        return jsonify({"deleted": deleted, "tool_name": tool_name, "key": key}), 200

    except Exception as e:
        logger.error(f"[REST API] tools config DELETE error: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete config key"}), 500


@tools_bp.route("/tools/<tool_name>/test", methods=["POST"])
@require_session
def test_tool(tool_name: str):
    """Validate that all required secret config fields are stored for a tool."""
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service

        registry = ToolRegistryService()
        if tool_name not in registry.tools:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        schema = registry.get_tool_config_schema(tool_name)
        db = get_shared_db_service()
        config = ToolConfigService(db).get_tool_config(tool_name)

        missing = [k for k, v in schema.items() if v.get("required") and k not in config]
        if missing:
            return jsonify({"ok": False, "message": f"Missing required config: {missing}"}), 200

        return jsonify({"ok": True, "message": "Configuration looks complete"}), 200

    except Exception as e:
        logger.error(f"[REST API] tools test error: {e}", exc_info=True)
        return jsonify({"error": "Failed to test tool"}), 500


# ------------------------------------------------------------------
# OAuth2 endpoints — generic, tool-agnostic
# ------------------------------------------------------------------

@tools_bp.route("/tools/<tool_name>/oauth/start", methods=["GET"])
@require_session
def oauth_start(tool_name: str):
    """Generate OAuth2 authorization URL for a tool.

    Returns {"auth_url": "...", "state": "..."}.
    """
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.oauth_service import OAuthService

        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if not tool:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        manifest_auth = tool["manifest"].get("auth")
        if not manifest_auth or manifest_auth.get("type") != "oauth2":
            return jsonify({"error": f"Tool '{tool_name}' does not use OAuth2"}), 400

        # Build redirect URI from request origin
        redirect_uri = request.args.get("redirect_uri")
        if not redirect_uri:
            # Derive from request host
            scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
            host = request.headers.get("X-Forwarded-Host", request.host)
            redirect_uri = f"{scheme}://{host}/tools/{tool_name}/oauth/callback"

        logger.info(
            f"[TOOLS API] OAuth start for '{tool_name}': "
            f"redirect_uri={redirect_uri} "
            f"X-Forwarded-Host={request.headers.get('X-Forwarded-Host', '(none)')} "
            f"X-Forwarded-Proto={request.headers.get('X-Forwarded-Proto', '(none)')} "
            f"Host={request.headers.get('Host', '(none)')}"
        )

        result = OAuthService().get_auth_url(tool_name, manifest_auth, redirect_uri)
        logger.info(
            f"[TOOLS API] OAuth start generated state={result.get('state', '?')[:16]}..."
        )
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"[TOOLS API] OAuth start error for '{tool_name}': {e}", exc_info=True)
        return jsonify({"error": "Failed to start OAuth flow"}), 500


@tools_bp.route("/tools/<tool_name>/oauth/callback", methods=["GET"])
def oauth_callback(tool_name: str):
    """OAuth2 callback — exchanges authorization code for tokens.

    No @require_session: the user arrives from an external redirect.
    CSRF protection via cryptographic state token validated against MemoryStore.

    On success, redirects to Brain admin with a success message.
    On error, redirects to Brain admin with an error message.
    """
    try:
        from services.oauth_service import OAuthService

        code = request.args.get("code")
        state = request.args.get("state")
        error = request.args.get("error")

        # Build Brain admin redirect URL
        scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
        host = request.headers.get("X-Forwarded-Host", request.host)
        brain_url = f"{scheme}://{host}/brain/"

        if error:
            error_desc = request.args.get("error_description", error)
            logger.warning(f"[TOOLS API] OAuth callback error for '{tool_name}': {error_desc}")
            from flask import redirect as flask_redirect
            return flask_redirect(f"{brain_url}?oauth_error={url_quote(error_desc)}&tool={tool_name}")

        if not code or not state:
            from flask import redirect as flask_redirect
            return flask_redirect(f"{brain_url}?oauth_error=Missing+code+or+state&tool={tool_name}")

        logger.info(
            f"[TOOLS API] OAuth callback for '{tool_name}': "
            f"state={state[:16]}... code={code[:12]}... "
            f"full_url={request.url[:200]}"
        )

        result = OAuthService().exchange_code(state, code)

        from flask import redirect as flask_redirect
        return flask_redirect(f"{brain_url}?oauth_success=true&tool={tool_name}")

    except ValueError as ve:
        logger.warning(f"[TOOLS API] OAuth callback validation error: {ve}")

        # Handle duplicate callback (browser double-fetch / redirect race).
        # If state was already consumed but tokens were stored by the first
        # call, treat this as a success rather than surfacing an error.
        if "expired" in str(ve).lower() or "invalid" in str(ve).lower():
            try:
                status = OAuthService().get_oauth_status(tool_name)
                if status.get("connected"):
                    logger.info(
                        f"[TOOLS API] OAuth callback duplicate for '{tool_name}' "
                        f"— state already consumed but tool is connected, treating as success"
                    )
                    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
                    host = request.headers.get("X-Forwarded-Host", request.host)
                    brain_url = f"{scheme}://{host}/brain/"
                    from flask import redirect as flask_redirect
                    return flask_redirect(f"{brain_url}?oauth_success=true&tool={tool_name}")
            except Exception:
                pass  # Fall through to normal error handling

        scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
        host = request.headers.get("X-Forwarded-Host", request.host)
        brain_url = f"{scheme}://{host}/brain/"
        from flask import redirect as flask_redirect
        return flask_redirect(f"{brain_url}?oauth_error={url_quote(str(ve)[:200])}&tool={tool_name}")
    except Exception as e:
        logger.error(f"[TOOLS API] OAuth callback error for '{tool_name}': {e}", exc_info=True)
        scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
        host = request.headers.get("X-Forwarded-Host", request.host)
        brain_url = f"{scheme}://{host}/brain/"
        from flask import redirect as flask_redirect
        return flask_redirect(f"{brain_url}?oauth_error=Internal+error&tool={tool_name}")


@tools_bp.route("/tools/<tool_name>/oauth/status", methods=["GET"])
@require_session
def oauth_status(tool_name: str):
    """Return OAuth connection status for a tool."""
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.oauth_service import OAuthService

        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if not tool:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        manifest_auth = tool["manifest"].get("auth")
        if not manifest_auth or manifest_auth.get("type") != "oauth2":
            return jsonify({"error": f"Tool '{tool_name}' does not use OAuth2"}), 400

        status = OAuthService().get_oauth_status(tool_name)
        status["provider_hint"] = manifest_auth.get("provider_hint", "")
        return jsonify(status), 200

    except Exception as e:
        logger.error(f"[TOOLS API] OAuth status error for '{tool_name}': {e}", exc_info=True)
        return jsonify({"error": "Failed to get OAuth status"}), 500


@tools_bp.route("/tools/<tool_name>/oauth/disconnect", methods=["POST"])
@require_session
def oauth_disconnect(tool_name: str):
    """Remove all OAuth tokens for a tool."""
    try:
        from services.tool_registry_service import ToolRegistryService
        from services.oauth_service import OAuthService

        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if not tool:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 404

        ok = OAuthService().disconnect(tool_name)
        if ok:
            return jsonify({"disconnected": True, "tool_name": tool_name}), 200
        else:
            return jsonify({"error": "Failed to disconnect"}), 500

    except Exception as e:
        logger.error(f"[TOOLS API] OAuth disconnect error for '{tool_name}': {e}", exc_info=True)
        return jsonify({"error": "Failed to disconnect OAuth"}), 500
