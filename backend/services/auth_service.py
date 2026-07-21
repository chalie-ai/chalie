"""Master-account authentication — the one login path.

Owns password verification against the ``master_account`` table, the vault
open that follows it, and the post-unlock capability reconnect. Both the REST
``/api/auth/login`` endpoint and the per-request pre-flight (:meth:`
AuthService.try_login`) call it, so there is exactly one way to log in.
"""

import json
import logging

from werkzeug.security import check_password_hash

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class AuthService:
    """Master-account login.

    Database, vault and capability imports are function-local: this module is
    imported while the API is being built, before those subsystems are wired.
    """

    def login(self, username: str, password: str) -> str | None:
        """Verify the master password and open the vault.

        Returns the resulting vault state — ``"unlocked"``, or ``"locked"``
        when the password was right but the vault would not open — or ``None``
        when the credentials do not match the master account. Callers map
        ``None`` to 401; a correct password against a shut vault is still a
        successful login and must never be reported as a bad one.

        The vault is opened with
        :meth:`~services.vault_service.VaultService.unlock_or_restore`, which
        rebuilds the live key from a filesystem backup when the stored row is
        missing or corrupt. If neither opens, the vault stays sealed and the
        master account is left exactly as it was — nothing here ever deletes
        or resets either of them.

        A database failure is not caught: an unreadable ``master_account`` is
        not a wrong password, and reporting it as one would hide the fault.
        """
        from services.database import Database
        from services.vault_service import (
            get_vault_service, OUTCOME_UNLOCKED, OUTCOME_RESTORED,
        )

        row = Database.conn().execute(
            "SELECT password_hash FROM master_account WHERE username = :username",
            {"username": username},
        ).fetchone()
        if not row or not check_password_hash(row[0], password):
            return None

        try:
            outcome = get_vault_service().unlock_or_restore(password)
        except Exception as exc:
            logger.error("[Auth] Vault open failed unexpectedly during login: %s", exc)
            return "locked"

        if outcome not in (OUTCOME_UNLOCKED, OUTCOME_RESTORED):
            logger.error(
                "[Auth] Vault did not open (%s) — the account and its key backups "
                "are left intact; this needs operator intervention.", outcome,
            )
            return "locked"

        self._reconnect_capabilities()
        return "unlocked"

    def try_login(self) -> bool:
        """Log in from a ``credentials.json`` at the install root, if one exists.

        Local-development convenience: the file holds ``{"username", "password"}``
        and takes the login page out of the loop. It is gitignored and absent
        from release builds, so on a normal install this whole call is a single
        filesystem stat.

        Does exactly one thing: read the file and attempt :meth:`login` with
        its credentials. Returns ``True`` only on a real unlock — ``login``
        also returns ``"locked"`` when the password is right but the vault
        will not open, a state no session can fix. Deciding *when* to attempt
        this is the request hook's job, not ours. It never raises — an
        exception here would break every endpoint at once.
        """
        path = FileMapperService.get_dev_credentials_path()
        if not path.is_file():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self.login(data["username"], data["password"]) == "unlocked"
        except Exception as exc:
            logger.error("[Auth] Auto-login from credentials.json failed: %s", exc)
            return False

    def vault_state(self) -> str:
        """Return the current vault state.

        All errors are swallowed so that a database or vault problem never
        crashes a status check.
        """
        try:
            from services.vault_service import get_vault_service
            return get_vault_service().get_state()
        except Exception as exc:
            logger.warning("[Auth] vault state check failed: %s", exc)
            return "uninitialized"

    def _reconnect_capabilities(self) -> None:
        """Load capability plugins and reconnect any that have stored credentials.

        Post-unlock hook: runs after vault unlock so capabilities can decrypt
        their stored credentials. Capability init is deferred from boot
        (``run.py``) to here because the vault requires the master password to
        unseal.

        Any error is caught and logged as a warning so a capability failure
        never prevents a successful login response from being returned.
        """
        try:
            from capabilities import load_capabilities

            capabilities = load_capabilities()
            reconnected = 0
            for cap in capabilities.values():
                if cap.connect():
                    reconnected += 1
            logger.info(
                "[Auth] Post-unlock capability reconnect: %d/%d capabilities active",
                reconnected,
                len(capabilities),
            )
        except Exception as exc:
            logger.warning("[Auth] Post-unlock capability reconnect failed: %s", exc)
