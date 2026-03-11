"""
Identity State Service — Zero-latency MemoryStore-backed identity authority.

Stores explicit identity fields (name, etc.) written synchronously when the user
makes an explicit identity statement (e.g., "call me Dylan"). Read by
FrontalCortexService before user_traits so identity is available immediately,
before the async memory-chunker pipeline has run.

MemoryStore key: identity_state
TTL: 7 days, refreshed on every write.
Schema per field:
    {
        "name": {
            "value": "Dylan",
            "normalized": "dylan",
            "display": "Dylan",
            "confidence": 0.95,
            "updated_at": 1708500000.0,
            "provisional": false,
            "previous": []
        }
    }
"""

import json
import logging
import time
from typing import Optional

from services.memory_client import MemoryClientService

logger = logging.getLogger(__name__)


class IdentityStateService:
    """Zero-latency MemoryStore-backed identity authority store."""

    _STORE_KEY_PREFIX = "identity_state"
    STORE_TTL = 604800          # 7 days
    MAX_PREVIOUS_HISTORY = 5

    def __init__(self):
        """Initialize the identity state service, setting the MemoryStore key."""
        self._store_key = self._STORE_KEY_PREFIX

    def set_field(
        self,
        field_name: str,
        value: str,
        confidence: float,
        provisional: bool = False,
    ) -> bool:
        """
        Store an identity field in MemoryStore.

        - Normalizes value to lowercase for dedup comparisons.
        - Stores display form as title-case when input is all-lowercase;
          otherwise preserves the user's casing (McDonald, O'Brien).
        - On value change: prepends old display value to previous[], capped at
          MAX_PREVIOUS_HISTORY.
        - Refreshes TTL on every write.

        Returns:
            bool: True if stored successfully, False on error. Never raises.
        """
        try:
            r = MemoryClientService.create_connection()

            # Read-modify-write
            raw = r.get(self._store_key)
            blob = json.loads(raw) if raw else {}

            normalized = value.lower()
            # Preserve mixed-case input (McDonald); only .title() for all-lowercase
            display = value.title() if value.islower() else value

            existing = blob.get(field_name, {})
            old_normalized = existing.get('normalized', '')
            previous = list(existing.get('previous', []))

            # Only prepend to previous[] when the value actually changes
            if old_normalized and old_normalized != normalized:
                old_display = existing.get('display') or existing.get('value', '')
                if old_display:
                    previous = [old_display] + previous
                    previous = previous[:self.MAX_PREVIOUS_HISTORY]

            blob[field_name] = {
                'value': display,
                'normalized': normalized,
                'display': display,
                'confidence': confidence,
                'updated_at': time.time(),
                'provisional': provisional,
                'previous': previous,
            }

            r.setex(self._store_key, self.STORE_TTL, json.dumps(blob))
            logger.debug(f"[IDENTITY STATE] Stored {field_name}='{display}'")
            return True

        except Exception as e:
            logger.warning(f"[IDENTITY STATE] set_field failed (non-fatal): {e}")
            return False

    def get_all(self) -> dict:
        """
        Retrieve the full identity state blob from MemoryStore.

        Returns:
            dict: Full blob mapping field names to their field dicts (each with
                  keys ``value``, ``normalized``, ``display``, ``confidence``,
                  ``updated_at``, ``provisional``, and ``previous``).  May also
                  include the reserved ``_onboarding`` tracking key.  Returns
                  ``{}`` on missing key or MemoryStore error.  Never raises.
        """
        try:
            r = MemoryClientService.create_connection()
            raw = r.get(self._store_key)
            if not raw:
                return {}
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[IDENTITY STATE] get_all failed (non-fatal): {e}")
            return {}

    def get_field(self, field_name: str) -> Optional[dict]:
        """
        Get a specific identity field.

        Returns:
            dict or None: Field data dict, or None if not found. Never raises.
        """
        try:
            blob = self.get_all()
            return blob.get(field_name)
        except Exception as e:
            logger.warning(f"[IDENTITY STATE] get_field failed (non-fatal): {e}")
            return None

    def clear_field(self, field_name: str) -> bool:
        """
        Remove a specific identity field from the blob.
        Other fields are unaffected.

        Returns:
            bool: True if successful, False on error. Never raises.
        """
        try:
            r = MemoryClientService.create_connection()
            raw = r.get(self._store_key)
            if not raw:
                return True
            blob = json.loads(raw)
            if field_name in blob:
                del blob[field_name]
                r.setex(self._store_key, self.STORE_TTL, json.dumps(blob))
            return True
        except Exception as e:
            logger.warning(f"[IDENTITY STATE] clear_field failed (non-fatal): {e}")
            return False

    def set_onboarding_state(self, onboarding_state: dict) -> bool:
        """
        Write the _onboarding tracking dict into the identity blob.

        Uses the same read-modify-write pattern as set_field.
        Existing identity fields are unaffected.

        Returns:
            bool: True if successful, False on error. Never raises.
        """
        try:
            r = MemoryClientService.create_connection()
            raw = r.get(self._store_key)
            blob = json.loads(raw) if raw else {}
            blob['_onboarding'] = onboarding_state
            r.setex(self._store_key, self.STORE_TTL, json.dumps(blob))
            return True
        except Exception as e:
            logger.warning(
                f"[IDENTITY STATE] set_onboarding_state failed (non-fatal): {e}"
            )
            return False
