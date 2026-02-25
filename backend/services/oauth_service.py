"""
OAuth Service — Generic OAuth2 authorization flow for any tool.

Tools opt into OAuth by declaring an `auth` block in their manifest.
This service handles the full flow: auth URL generation, code exchange,
token storage (via tool_configs reserved keys), and automatic refresh.

Never references any specific tool or provider by name.
"""

import hashlib
import json
import logging
import secrets
import time
import base64
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# Reserved keys stored in tool_configs for OAuth tokens
RESERVED_OAUTH_KEYS = frozenset({
    "_oauth_access_token",
    "_oauth_refresh_token",
    "_oauth_token_expires_at",
    "_oauth_connected_at",
    "_oauth_scopes",
})


class OAuthService:
    """Generic OAuth2 service — tool-agnostic, provider-agnostic."""

    # Redis key prefix for in-flight OAuth state (PKCE verifier, tool name)
    _REDIS_STATE_PREFIX = "oauth_state:"
    _STATE_TTL = 300  # 5 minutes

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Auth URL generation
    # ------------------------------------------------------------------

    def get_auth_url(
        self,
        tool_name: str,
        manifest_auth: dict,
        redirect_uri: str,
    ) -> dict:
        """
        Generate an OAuth2 authorization URL with PKCE and CSRF state.

        Returns:
            {
                "auth_url": str,
                "state": str,       # for CSRF validation
            }
        """
        auth_url = manifest_auth.get("authorization_url")
        scopes = manifest_auth.get("scopes", [])
        use_pkce = manifest_auth.get("pkce", True)
        extra_params = manifest_auth.get("extra_auth_params", {})

        if not auth_url:
            raise ValueError("manifest auth missing 'authorization_url'")

        # Retrieve client_id from tool config
        client_id = self._get_config_value(tool_name, "client_id")
        if not client_id:
            raise ValueError(f"Tool '{tool_name}' missing 'client_id' in config")

        # Generate CSRF state token
        state = secrets.token_urlsafe(32)

        # PKCE: generate code_verifier and code_challenge
        code_verifier = None
        if use_pkce:
            code_verifier = secrets.token_urlsafe(64)[:128]

        # Store state → {tool_name, code_verifier} in Redis
        self._store_oauth_state(state, {
            "tool_name": tool_name,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
        })

        # Build authorization URL
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
            **extra_params,
        }

        if use_pkce and code_verifier:
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).rstrip(b"=").decode()
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        full_url = f"{auth_url}?{urlencode(params)}"

        return {
            "auth_url": full_url,
            "state": state,
        }

    # ------------------------------------------------------------------
    # Code exchange
    # ------------------------------------------------------------------

    def exchange_code(
        self,
        state: str,
        code: str,
    ) -> dict:
        """
        Exchange an authorization code for tokens.

        Validates state token against Redis, retrieves PKCE verifier,
        exchanges code at the token endpoint, and stores tokens.

        Returns:
            {"tool_name": str, "connected": True, "scopes": [...]}

        Raises:
            ValueError on invalid state or exchange failure.
        """
        # Retrieve and validate state
        state_data = self._pop_oauth_state(state)
        if not state_data:
            raise ValueError("Invalid or expired OAuth state token")

        tool_name = state_data["tool_name"]
        code_verifier = state_data.get("code_verifier")
        redirect_uri = state_data.get("redirect_uri", "")

        # Load manifest auth config
        manifest_auth = self._get_manifest_auth(tool_name)
        if not manifest_auth:
            raise ValueError(f"Tool '{tool_name}' has no auth config in manifest")

        token_url = manifest_auth.get("token_url")
        if not token_url:
            raise ValueError("manifest auth missing 'token_url'")

        # Get client credentials from tool config
        client_id = self._get_config_value(tool_name, "client_id")
        client_secret = self._get_config_value(tool_name, "client_secret")
        if not client_id or not client_secret:
            raise ValueError(f"Tool '{tool_name}' missing client_id or client_secret")

        # Build token request
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if code_verifier:
            token_data["code_verifier"] = code_verifier

        # Exchange code for tokens
        try:
            resp = requests.post(token_url, data=token_data, timeout=15)
            resp.raise_for_status()
            tokens = resp.json()
        except requests.RequestException as e:
            logger.error(f"[OAUTH] Token exchange failed for '{tool_name}': {e}")
            raise ValueError(f"Token exchange failed: {str(e)[:200]}")

        if "error" in tokens:
            error_desc = tokens.get("error_description", tokens["error"])
            raise ValueError(f"Token exchange error: {error_desc}")

        # Store tokens
        self._store_tokens(tool_name, tokens, manifest_auth)

        scopes = tokens.get("scope", " ".join(manifest_auth.get("scopes", []))).split()
        return {
            "tool_name": tool_name,
            "connected": True,
            "scopes": scopes,
        }

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def refresh_if_needed(
        self,
        tool_name: str,
        manifest_auth: dict,
        margin_seconds: int = 300,
    ) -> str | None:
        """
        Check if the access token is expired or about to expire.
        If so, refresh it and return the new access token.
        If not expired, return the current access token.
        Returns None if no OAuth tokens exist.
        """
        config = self._get_all_config(tool_name)

        access_token = config.get("_oauth_access_token")
        if not access_token:
            return None

        refresh_token = config.get("_oauth_refresh_token")
        expires_at_str = config.get("_oauth_token_expires_at", "0")

        try:
            expires_at = float(expires_at_str)
        except (ValueError, TypeError):
            expires_at = 0

        # If token is still valid, return it
        if time.time() + margin_seconds < expires_at:
            return access_token

        # Need refresh
        if not refresh_token:
            logger.warning(f"[OAUTH] Token expired for '{tool_name}' but no refresh token available")
            return access_token  # Return stale token; tool will get a 401

        token_url = manifest_auth.get("token_url")
        if not token_url:
            logger.warning(f"[OAUTH] No token_url in manifest for '{tool_name}'")
            return access_token

        client_id = self._get_config_value(tool_name, "client_id")
        client_secret = self._get_config_value(tool_name, "client_secret")
        if not client_id or not client_secret:
            logger.warning(f"[OAUTH] Missing client credentials for '{tool_name}'")
            return access_token

        try:
            resp = requests.post(token_url, data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            }, timeout=15)
            resp.raise_for_status()
            tokens = resp.json()
        except requests.RequestException as e:
            logger.error(f"[OAUTH] Token refresh failed for '{tool_name}': {e}")
            return access_token  # Return stale token

        if "error" in tokens:
            logger.error(f"[OAUTH] Token refresh error for '{tool_name}': {tokens.get('error_description', tokens['error'])}")
            return access_token

        # Store refreshed tokens
        self._store_tokens(tool_name, tokens, manifest_auth)
        logger.info(f"[OAUTH] Token refreshed for '{tool_name}'")

        return tokens.get("access_token", access_token)

    # ------------------------------------------------------------------
    # Status & disconnect
    # ------------------------------------------------------------------

    def get_oauth_status(self, tool_name: str) -> dict:
        """Return OAuth connection status for a tool."""
        config = self._get_all_config(tool_name)
        connected = bool(config.get("_oauth_access_token"))
        return {
            "connected": connected,
            "connected_at": config.get("_oauth_connected_at", ""),
            "scopes": config.get("_oauth_scopes", "").split() if connected else [],
        }

    def disconnect(self, tool_name: str) -> bool:
        """Remove all OAuth tokens for a tool."""
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            config_svc = ToolConfigService(db)

            for key in RESERVED_OAUTH_KEYS:
                config_svc.delete_tool_config_key(tool_name, key)

            logger.info(f"[OAUTH] Disconnected OAuth for '{tool_name}'")
            return True
        except Exception as e:
            logger.error(f"[OAUTH] Disconnect failed for '{tool_name}': {e}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_tokens(self, tool_name: str, tokens: dict, manifest_auth: dict):
        """Store OAuth tokens in tool_configs as reserved keys."""
        try:
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            now = str(int(time.time()))

            # Calculate expiry timestamp
            expires_in = tokens.get("expires_in")
            expires_at = str(int(time.time() + int(expires_in))) if expires_in else "0"

            # Prepare token data
            token_data = {
                "_oauth_access_token": tokens["access_token"],
                "_oauth_token_expires_at": expires_at,
                "_oauth_connected_at": now,
                "_oauth_scopes": tokens.get("scope", " ".join(manifest_auth.get("scopes", []))),
            }

            # Only update refresh token if a new one was provided
            if tokens.get("refresh_token"):
                token_data["_oauth_refresh_token"] = tokens["refresh_token"]

            # Write directly to DB, bypassing reserved-key guard
            with db.connection() as conn:
                cursor = conn.cursor()
                for key, value in token_data.items():
                    cursor.execute(
                        """
                        INSERT INTO tool_configs (tool_name, config_key, config_value)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (tool_name, config_key)
                        DO UPDATE SET config_value = EXCLUDED.config_value,
                                      updated_at = NOW()
                        """,
                        (tool_name, key, str(value))
                    )
                cursor.close()

            logger.info(f"[OAUTH] Stored tokens for '{tool_name}'")
        except Exception as e:
            logger.error(f"[OAUTH] Failed to store tokens for '{tool_name}': {e}", exc_info=True)
            raise

    def _get_config_value(self, tool_name: str, key: str) -> str | None:
        """Get a single config value for a tool."""
        config = self._get_all_config(tool_name)
        return config.get(key)

    def _get_all_config(self, tool_name: str) -> dict:
        """Get all config for a tool."""
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            return ToolConfigService(get_shared_db_service()).get_tool_config(tool_name)
        except Exception:
            return {}

    def _get_manifest_auth(self, tool_name: str) -> dict | None:
        """Get auth config from a tool's manifest."""
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            tool = registry.tools.get(tool_name)
            if tool:
                return tool["manifest"].get("auth")
        except Exception:
            pass
        return None

    def _store_oauth_state(self, state: str, data: dict):
        """Store OAuth state data in Redis with TTL."""
        try:
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()
            redis.setex(
                f"{self._REDIS_STATE_PREFIX}{state}",
                self._STATE_TTL,
                json.dumps(data),
            )
        except Exception as e:
            logger.error(f"[OAUTH] Failed to store state in Redis: {e}")
            raise ValueError("Failed to initiate OAuth flow (Redis unavailable)")

    def _pop_oauth_state(self, state: str) -> dict | None:
        """Retrieve and delete OAuth state data from Redis."""
        try:
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()
            key = f"{self._REDIS_STATE_PREFIX}{state}"
            data = redis.get(key)
            if data:
                redis.delete(key)
                return json.loads(data)
        except Exception as e:
            logger.error(f"[OAUTH] Failed to retrieve state from Redis: {e}")
        return None
