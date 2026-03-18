"""
HTTP client for Chalie backend API.

The dashboard authenticates with Chalie using a wrapper bearer token
that is created during bootstrap and stored in the dashboard DB.
"""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds


class ChalieClient:
    """Thin HTTP wrapper for talking to Chalie's backend API."""

    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # ── Signals ────────────────────────────────────────────────────────────

    def push_signal(
        self,
        signal_type: str,
        content: str,
        source: str | None = None,
        activation_energy: float = 0.5,
        metadata: dict | None = None,
    ) -> dict:
        """Push a signal to Chalie's world state.

        Returns the JSON response body on success, or an error dict.
        """
        body: dict[str, Any] = {
            "signal_type": signal_type,
            "content": content,
            "activation_energy": activation_energy,
        }
        if source:
            body["source"] = source
        if metadata:
            body["metadata"] = metadata

        try:
            resp = requests.post(
                f"{self.base_url}/api/signals",
                json=body,
                headers=self._headers(),
                timeout=_TIMEOUT,
            )
            return resp.json()
        except Exception as e:
            logger.warning("[ChalieClient] push_signal failed: %s", e)
            return {"error": str(e)}

    # ── Context ────────────────────────────────────────────────────────────

    def get_context(self) -> dict:
        """Fetch the user's current ambient context.

        Returns a dict with location, timezone, device, energy, attention
        fields (whichever Chalie has available).
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/context",
                headers=self._headers(),
                timeout=_TIMEOUT,
            )
            if resp.ok:
                return resp.json()
            return {}
        except Exception as e:
            logger.warning("[ChalieClient] get_context failed: %s", e)
            return {}

    # ── Interface registration ─────────────────────────────────────────────

    def register_interface(self, name: str, host: str, port: int) -> dict:
        """Register an interface with Chalie so its tools appear in ACT loop.

        Uses a pairing key to authenticate (not bearer token).
        """
        # First generate a pairing key
        try:
            resp = requests.post(
                f"{self.base_url}/api/interfaces/pairing-key",
                headers=self._headers(),
                timeout=_TIMEOUT,
            )
            if not resp.ok:
                return {"error": f"Failed to generate pairing key: {resp.status_code}"}
            key_data = resp.json()
            pairing_key = key_data.get("pairing_key")
        except Exception as e:
            return {"error": f"Failed to generate pairing key: {e}"}

        # Complete pairing
        try:
            resp = requests.post(
                f"{self.base_url}/api/interfaces/pair",
                json={
                    "pairing_key": pairing_key,
                    "name": name,
                    "host": host,
                    "port": port,
                },
                timeout=_TIMEOUT,
            )
            if resp.ok:
                return resp.json()
            return {"error": f"Pairing failed: {resp.status_code} {resp.text}"}
        except Exception as e:
            return {"error": f"Pairing failed: {e}"}

    def unregister_interface(self, chalie_interface_id: str) -> bool:
        """Remove an interface from Chalie's registry."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/interfaces/{chalie_interface_id}",
                headers=self._headers(),
                timeout=_TIMEOUT,
            )
            return resp.ok
        except Exception as e:
            logger.warning("[ChalieClient] unregister failed: %s", e)
            return False

    def refresh_capabilities(self, chalie_interface_id: str) -> dict:
        """Re-fetch capabilities from interface via Chalie."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/interfaces/{chalie_interface_id}/refresh",
                headers=self._headers(),
                timeout=_TIMEOUT,
            )
            if resp.ok:
                return resp.json()
            return {"error": f"Refresh failed: {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    # ── Bootstrap ──────────────────────────────────────────────────────────

    def create_wrapper_token(self, session_cookie: str) -> str | None:
        """Create a wrapper token for the dashboard using session auth.

        This is called once during bootstrap — the user must be logged in.
        Returns the raw token string, or None on failure.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/wrappers",
                json={
                    "name": "Dashboard Gateway",
                    "capabilities": {"signals": ["*"]},
                    "permissions": {
                        "query": ["memory", "context"],
                        "update": ["context"],
                        "broadcast": False,
                    },
                    "metadata": {"type": "dashboard", "version": "1.0.0"},
                },
                cookies={"chalie_session": session_cookie},
                timeout=_TIMEOUT,
            )
            if resp.ok:
                data = resp.json()
                return data.get("token")
            logger.error("[ChalieClient] Failed to create wrapper token: %s", resp.text)
        except Exception as e:
            logger.error("[ChalieClient] Bootstrap failed: %s", e)
        return None

    # ── Health ─────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Check if Chalie backend is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/ready", timeout=5)
            return resp.ok
        except Exception:
            return False
