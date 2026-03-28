"""
Unit tests for the capability framework base layer.

Covers:
- :class:`capabilities.base.AbstractCapability` — abstract interface,
  concrete stub instantiation, connection state, and credential helpers.
- :func:`capabilities.load_capabilities` — filesystem-based discovery.
- :func:`capabilities.caldav_capability.providers.resolve_provider` — provider
  lookup including unknown name handling.
- ``capabilities/caldav_capability/manifest.yaml`` — required manifest fields.
- :data:`capabilities.caldav_capability.providers.PROVIDERS` — all 6 expected
  provider entries are present.

All tests use ``@pytest.mark.unit`` and perform no real network or file-system
I/O (database interactions are mocked via in-process helpers).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockVaultService:
    """In-memory stand-in for :class:`services.vault_service.VaultService`.

    Uses AES-256-GCM with a deterministic 32-byte test key so that
    ``store_credential`` / ``load_credential`` round-trips work correctly
    during tests without requiring a real database or registered user.

    Methods mirror the public surface called by the credential helpers.
    """

    def __init__(self) -> None:
        """Initialise the mock vault with a fixed 32-byte test key."""
        import os
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._key = b"T" * 32  # deterministic test key
        self._aesgcm = AESGCM(self._key)

    def encrypt_str(self, s: str) -> bytes:
        """UTF-8–encode *s* and encrypt with AES-256-GCM.

        Args:
            s: Plaintext string to encrypt.

        Returns:
            bytes: ``nonce (12 B) + ciphertext + tag (16 B)``.
        """
        import os
        nonce = os.urandom(12)
        ciphertext_tag = self._aesgcm.encrypt(nonce, s.encode("utf-8"), None)
        return nonce + ciphertext_tag

    def decrypt_str(self, blob: bytes) -> str:
        """Decrypt *blob* and UTF-8–decode the result.

        Args:
            blob: Encrypted blob as produced by :meth:`encrypt_str`.

        Returns:
            str: Decrypted plaintext string.
        """
        nonce = blob[:12]
        ciphertext_tag = blob[12:]
        plaintext = self._aesgcm.decrypt(nonce, ciphertext_tag, None)
        return plaintext.decode("utf-8")


class _InMemoryToolConfigService:
    """In-memory stand-in for :class:`services.tool_config_service.ToolConfigService`.

    Stores tool configurations in a plain :class:`dict` so that
    ``store_credential`` / ``load_credential`` round-trip correctly during
    tests without requiring a real database.

    Methods intentionally mirror the public surface of the real service.
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory store."""
        self._store: dict[str, dict] = {}

    def set_tool_config(self, tool_name: str, config: dict) -> None:
        """Upsert *config* entries for *tool_name*.

        Args:
            tool_name: Capability / tool identifier used as the top-level key.
            config:    Mapping of ``config_key`` → ``encrypted_value`` to upsert.
        """
        if tool_name not in self._store:
            self._store[tool_name] = {}
        self._store[tool_name].update(config)

    def get_tool_config(self, tool_name: str) -> dict:
        """Return all config entries for *tool_name*, or an empty dict.

        Args:
            tool_name: Capability / tool identifier.

        Returns:
            dict: Config key → value mapping; empty dict if not found.
        """
        return dict(self._store.get(tool_name, {}))

    def delete_tool_config(self, tool_name: str) -> None:
        """Remove all entries for *tool_name*.

        Args:
            tool_name: Capability / tool identifier.
        """
        self._store.pop(tool_name, None)


class _MinimalCapability:
    """Minimal concrete subclass of :class:`capabilities.base.AbstractCapability`.

    All 7 abstract methods are implemented with trivial stubs so that the
    class can be instantiated during tests.  No network or I/O occurs.
    """

    # Deferred import so the fixture environment is set up before the class
    # tries to resolve ABC machinery.
    @staticmethod
    def _build():
        """Dynamically build and return an instance of the concrete stub.

        Using a factory avoids importing ``capabilities.base`` at module
        collection time, which could trigger service imports before mocks are
        in place.

        Returns:
            _MinimalCapabilityImpl: A freshly instantiated concrete capability.
        """
        from capabilities.base import AbstractCapability

        class _MinimalCapabilityImpl(AbstractCapability):
            """Stub implementation of all 7 abstract methods."""

            def get_id(self) -> str:
                """Return the stub capability identifier.

                Returns:
                    str: Always ``"stub"``.
                """
                return "stub"

            def get_manifest(self) -> dict:
                """Return a minimal manifest dict.

                Returns:
                    dict: Minimal manifest with ``id`` and ``entry_class``.
                """
                return {"id": "stub", "entry_class": "_MinimalCapabilityImpl"}

            def configure(self, credentials: dict) -> None:
                """No-op configure stub.

                Args:
                    credentials: Ignored.
                """

            def connect(self) -> bool:
                """No-op connect stub that always returns ``False``.

                Returns:
                    bool: Always ``False``.
                """
                return False

            def disconnect(self) -> None:
                """No-op disconnect stub."""

            def ingest(self) -> list:
                """No-op ingest stub that returns an empty list.

                Returns:
                    list: Empty list.
                """
                return []

            def understand(self, items: list) -> list:
                """No-op understand stub that returns items unchanged."""
                return items

            def monitor(self) -> None:
                """No-op monitor stub."""

            def act(self, action: str, params: dict) -> dict:
                """No-op act stub."""
                return {"success": True}

            def get_tools(self) -> list:
                """No-op get_tools stub that returns an empty list.

                Returns:
                    list: Empty list.
                """
                return []

        return _MinimalCapabilityImpl()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAbstractCapability:
    """Tests for the :class:`capabilities.base.AbstractCapability` interface."""

    def test_abstract_capability_cannot_be_instantiated(self):
        """AbstractCapability must raise TypeError when instantiated directly.

        Verifies that the ABC mechanism correctly prevents direct
        instantiation of the abstract base class.
        """
        from capabilities.base import AbstractCapability

        with pytest.raises(TypeError):
            AbstractCapability()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        """A concrete subclass implementing all 7 abstract methods must instantiate.

        Verifies that ``_MinimalCapability._build()`` produces a valid object
        and that the result is an instance of ``AbstractCapability``.
        """
        from capabilities.base import AbstractCapability

        cap = _MinimalCapability._build()
        assert isinstance(cap, AbstractCapability)

    def test_is_connected_defaults_to_false(self):
        """``is_connected()`` must return ``False`` on a freshly created instance.

        The ``_connected`` flag is set to ``False`` in ``AbstractCapability.__init__``
        and must not be mutated by instantiation alone.
        """
        cap = _MinimalCapability._build()
        assert cap.is_connected() is False

    def test_store_and_load_credential_roundtrip(self):
        """``store_credential`` followed by ``load_credential`` must return the original value.

        Patches ``services.vault_service.get_vault_service`` to return a
        deterministic in-memory vault mock and
        ``capabilities.base._get_tool_config_service`` to return an in-memory
        service, ensuring no real database or unlocked vault is required.
        """
        mem_svc = _InMemoryToolConfigService()
        mock_vault = _MockVaultService()

        with patch("services.vault_service.get_vault_service", return_value=mock_vault), \
             patch("capabilities.base._get_tool_config_service", return_value=mem_svc):

            cap = _MinimalCapability._build()
            cap.store_credential("username", "alice")
            result = cap.load_credential("username")

        assert result == "alice"

    def test_load_credential_returns_none_for_missing_key(self):
        """``load_credential`` must return ``None`` when the key does not exist.

        The in-memory service returns an empty dict for all ``get_tool_config``
        calls, simulating the absence of any stored credential.
        """
        mem_svc = _InMemoryToolConfigService()
        mock_vault = _MockVaultService()

        with patch("services.vault_service.get_vault_service", return_value=mock_vault), \
             patch("capabilities.base._get_tool_config_service", return_value=mem_svc):

            cap = _MinimalCapability._build()
            result = cap.load_credential("missing")

        assert result is None


def _make_health_cap(monitor_effect=None):
    """Build a connected stub capability with optional monitor side effect."""
    cap = _MinimalCapability._build()
    cap._connected = True
    if monitor_effect is not None:
        cap.monitor = MagicMock(side_effect=monitor_effect)
    return cap


@pytest.mark.unit
class TestHealthTracking:
    """Tests for capability health tracking via ``run_monitor()``."""

    def test_health_details_defaults(self):
        """Fresh capability has zero errors and no last_error."""
        cap = _MinimalCapability._build()
        assert cap.health_details() == {"connected": False, "error_count": 0, "last_error": None}

    def test_run_monitor_success_resets_and_persists(self):
        """Successful monitor() resets error count and persists zero."""
        mem_svc = _InMemoryToolConfigService()
        cap = _make_health_cap()
        cap._error_count = 3
        cap._last_error = "old"
        with patch("capabilities.base._get_tool_config_service", return_value=mem_svc):
            cap.run_monitor()
        assert cap._error_count == 0
        assert mem_svc.get_tool_config("stub")["stub:error_count"] == "0"

    def test_run_monitor_increments_on_failure(self):
        """Failed monitor() increments error count and records the error."""
        cap = _make_health_cap(ConnectionError("server unreachable"))
        with patch("capabilities.base._get_tool_config_service", return_value=_InMemoryToolConfigService()):
            cap.run_monitor()
        assert cap.health_details() == {"connected": True, "error_count": 1, "last_error": "server unreachable"}

    def test_run_monitor_auto_disconnects_after_threshold(self):
        """Capability auto-disconnects after MAX_CONSECUTIVE_FAILURES."""
        cap = _make_health_cap(ConnectionError("down"))
        cap._error_count = cap.MAX_CONSECUTIVE_FAILURES - 1  # one away
        with patch("capabilities.base._get_tool_config_service", return_value=_InMemoryToolConfigService()):
            cap.run_monitor()  # this pushes it over
        assert cap._connected is False

    def test_run_monitor_persists_to_tool_configs(self):
        """Error state is persisted to tool_configs after each call."""
        mem_svc = _InMemoryToolConfigService()
        cap = _make_health_cap(RuntimeError("timeout"))
        with patch("capabilities.base._get_tool_config_service", return_value=mem_svc):
            cap.run_monitor()
        stored = mem_svc.get_tool_config("stub")
        assert stored["stub:error_count"] == "1"
        assert stored["stub:last_error"] == "timeout"

    def test_run_monitor_recovery_resets_count(self):
        """A successful monitor() after failures resets the counter."""
        cap = _make_health_cap()
        cap._error_count = 3
        cap._last_error = "was broken"
        with patch("capabilities.base._get_tool_config_service", return_value=_InMemoryToolConfigService()):
            cap.run_monitor()
        assert cap._error_count == 0
        assert cap._connected is True


@pytest.mark.unit
class TestLoadCapabilities:
    """Tests for :func:`capabilities.load_capabilities`."""

    def test_load_capabilities_returns_empty_dict_with_no_dirs(self):
        """``load_capabilities()`` must return ``{}`` when no sub-directories exist.

        Patches ``capabilities.Path`` so that the ``capabilities/`` directory
        scan yields an empty iterator, simulating a clean state with no
        capability packages installed.
        """
        from capabilities import load_capabilities

        mock_parent = MagicMock()
        mock_parent.iterdir.return_value = []

        mock_file_path = MagicMock()
        mock_file_path.parent = mock_parent

        with patch("capabilities.Path", return_value=mock_file_path):
            result = load_capabilities()

        assert result == {}


@pytest.mark.unit
class TestProviders:
    """Tests for :mod:`capabilities.caldav_capability.providers`."""

    def test_resolve_provider_returns_none_for_unknown(self):
        """``resolve_provider`` must return ``None`` for unrecognised names.

        Verifies that querying with a provider name that is not in the
        :data:`~capabilities.caldav_capability.providers.PROVIDERS` registry
        returns ``None`` rather than raising.
        """
        from capabilities.caldav_capability.providers import resolve_provider

        assert resolve_provider("nonexistent") is None

    def test_resolve_provider_returns_config_for_known_provider(self):
        """``resolve_provider('google')`` must return a config dict.

        Spot-checks that the returned dict includes a ``url`` starting with
        ``https://``, confirming that the registry entry is well-formed.
        """
        from capabilities.caldav_capability.providers import resolve_provider

        provider = resolve_provider("google")
        assert provider is not None
        assert provider["url"].startswith("https://")

    def test_resolve_provider_is_case_insensitive(self):
        """``resolve_provider`` must accept mixed-case provider names.

        Verifies that ``resolve_provider('Google')`` and
        ``resolve_provider('GOOGLE')`` both resolve successfully.
        """
        from capabilities.caldav_capability.providers import resolve_provider

        assert resolve_provider("Google") is not None
        assert resolve_provider("GOOGLE") is not None

    def test_providers_all_present(self):
        """All 6 expected provider names must be present in :data:`PROVIDERS`.

        This guards against accidental deletions from the registry and confirms
        that the six providers documented in the manifest are all wired up.
        """
        from capabilities.caldav_capability.providers import PROVIDERS

        expected = {"google", "apple", "fastmail", "nextcloud", "synology", "radicale"}
        assert expected.issubset(set(PROVIDERS.keys()))


@pytest.mark.unit
class TestManifest:
    """Tests for the ``caldav_capability`` manifest.yaml file."""

    def test_manifest_loads_correctly(self):
        """The manifest must parse successfully with required fields present.

        Loads the actual ``manifest.yaml`` from the repository (no mocking
        required as it is a static file).  Asserts that ``id == 'caldav'``
        and that ``entry_class == 'CaldavCapability'``.
        """
        manifest_path = (
            Path(__file__).parent.parent
            / "capabilities"
            / "caldav_capability"
            / "manifest.yaml"
        )
        assert manifest_path.exists(), f"manifest.yaml not found at {manifest_path}"

        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh)

        assert manifest["id"] == "caldav"
        assert manifest["entry_class"] == "CaldavCapability"
