"""
User authentication namespace — /auth endpoints for master account.
"""

import logging
from typing import TYPE_CHECKING, cast

from flask import request, jsonify
from flask_restx import Namespace, Resource
from services.database_service import text
from .auth import require_auth, _cookie_only, internal_only
from services.feature_flags import internal_dev_enabled
from werkzeug.security import generate_password_hash, check_password_hash

if TYPE_CHECKING:
    from services.wrapper_rate_limiter import WrapperRateLimiter

logger = logging.getLogger(__name__)

user_auth_bp = Namespace('user_auth', description='Master account authentication', path='/auth')


_LOGIN_RATE_LIMIT = 10   # attempts
_LOGIN_RATE_WINDOW = 60  # seconds


def _get_login_rate_limiter() -> "WrapperRateLimiter":
    from services.wrapper_rate_limiter import WrapperRateLimiter
    return WrapperRateLimiter(
        limit=_LOGIN_RATE_LIMIT, window_seconds=_LOGIN_RATE_WINDOW
    )


def _reconnect_capabilities() -> None:
    """Load capability plugins and reconnect any that have stored credentials.

    Post-unlock hook: runs after vault unlock so capabilities can decrypt their
    stored credentials.  Capability init is deferred from boot (``run.py``) to
    here because the vault requires the master password to unseal.

    Any error is caught and logged as a warning so a capability failure never
    prevents a successful login response from being returned to the client.
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


def _get_vault_state() -> str:
    """Return the current vault state via VaultService.get_state().

    All errors are swallowed so that a database problem never crashes
    the ``/auth/status`` endpoint.
    """
    try:
        from services.vault_service import get_vault_service
        return get_vault_service().get_state()
    except Exception as exc:
        logger.warning("[Auth] vault state check failed: %s", exc)
        return "uninitialized"


@user_auth_bp.route('/status')
class AuthStatusResource(Resource):
    @user_auth_bp.response(200, "Auth status")
    @user_auth_bp.response(500, "Failed to check auth status")
    def get(self):
        """Check whether master account exists, providers are configured, and
        user has session.

        Returns a JSON body with the following keys:

        * ``has_master_account`` — ``True`` if at least one row exists in
          ``master_account``.
        * ``has_providers``      — ``True`` if at least one active provider is configured.
        * ``has_session``        — ``True`` if the request carries a valid session token.
        * ``vault_state``        — ``"unlocked" | "locked" | "uninitialized"``.
        * ``internal_dev``       — ``True`` when in-development features are enabled.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.auth_session_service import validate_session

            db = get_shared_db_service()

            # Check master account
            with db.get_session() as session:
                account_count = cast("tuple[int, ...]", session.execute(
                    text("SELECT COUNT(*) FROM master_account")
                ).fetchone())[0]

            # Check providers (count only — avoids decryption which can fail if key changed)
            with db.get_session() as session:
                provider_count = cast("tuple[int, ...]", session.execute(
                    text("SELECT COUNT(*) FROM providers")
                ).fetchone())[0]

            # Check session — if the vault is sealed (server restarted while
            # session cookie survived), treat the session as invalid so every
            # frontend redirects to the login page, which re-unlocks the vault.
            has_session = validate_session(request)
            vault_state = _get_vault_state()
            if has_session and vault_state == "locked":
                has_session = False

            # Vision availability — gates the image-upload affordance in the chat UI.
            try:
                from services.database_service import get_shared_db_service
                from services.provider_db_service import ProviderDbService
                has_vision = (
                    ProviderDbService(get_shared_db_service()).get_vision_provider()
                    is not None
                )
            except Exception:
                has_vision = False

            return {
                "has_master_account": account_count > 0,
                "has_providers": provider_count > 0,
                "has_session": has_session,
                "vault_state": vault_state,
                "has_vision_provider": has_vision,
                "internal_dev": internal_dev_enabled(),
            }, 200
        except Exception as e:
            logger.error(f"[REST API] Auth status error: {e}")
            return {"error": "Failed to check auth status"}, 500


@user_auth_bp.route('/username')
class UsernameResource(Resource):
    @internal_only
    @require_auth
    @_cookie_only
    @user_auth_bp.response(200, "Username")
    @user_auth_bp.response(404, "No master account")
    @user_auth_bp.response(500, "Failed to read username")
    def get(self):
        """Return the master account LOGIN username for the authenticated dashboard
        session — the credential the device's UnlockVault screen submits to
        POST /auth/login. Cookie-session only; a wrapper bearer must not read it.
        """
        try:
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            with db.get_session() as session:
                row = session.execute(
                    text("SELECT username FROM master_account LIMIT 1")
                ).fetchone()
            if not row:
                return {"error": "No master account"}, 404
            return {"username": row[0]}, 200
        except Exception:
            logger.exception("[REST API] Get username error")
            return {"error": "Failed to read username"}, 500


@user_auth_bp.route('/register')
class RegisterResource(Resource):
    @user_auth_bp.response(201, "Account created")
    @user_auth_bp.response(400, "Validation error")
    @user_auth_bp.response(409, "Master account already exists")
    @user_auth_bp.response(500, "Failed to create account")
    def post(self):
        """Create master account. Fails (409) if one exists. Sets session cookie on success.

        On success the vault is initialised with the master password and immediately
        unlocked so that the caller receives ``vault_state: "unlocked"`` in the
        response body.  If vault initialisation fails after the account row has been
        committed the endpoint returns 500 — the account exists but the vault must be
        re-initialised (this scenario is logged clearly for operator investigation).
        """
        try:
            from services.database_service import get_shared_db_service
            from services.auth_session_service import create_session
            from services.vault_service import get_vault_service
            from flask import make_response

            data = request.get_json() or {}
            username = (data.get('username') or '').strip()
            password = data.get('password') or ''

            # Validation
            if not username:
                return {"error": "Username required"}, 400
            if len(password) < 8:
                return {"error": "Password must be at least 8 characters"}, 400

            db = get_shared_db_service()

            # Check if master account already exists
            with db.get_session() as session:
                existing = cast("tuple[int, ...]", session.execute(
                    text("SELECT COUNT(*) FROM master_account")
                ).fetchone())[0]

                if existing > 0:
                    return {"error": "Master account already exists"}, 409

                # Hash password and create account
                password_hash = generate_password_hash(password)
                session.execute(
                    text(
                        "INSERT INTO master_account "
                        "(username, password_hash) "
                        "VALUES (:username, :password_hash)"
                    ),
                    {"username": username, "password_hash": password_hash}
                )
                session.commit()

            # Initialise and immediately unlock the vault with the master password.
            # Runs after commit so a vault failure does not orphan a half-written row.
            vault = get_vault_service()
            try:
                vault.initialize(password)
                vault.unlock(password)
            except Exception as vault_exc:
                logger.error(
                    "[Auth] Vault initialisation failed after registration: %s", vault_exc
                )
                return {
                    "error": "Account created but vault "
                    "initialization failed"
                }, 500

            # Create session and set cookie
            resp = make_response(jsonify({"ok": True, "vault_state": "unlocked"}), 201)
            create_session(resp)
            return resp
        except Exception as e:
            logger.error(f"[REST API] Register error: {e}")
            return {"error": "Failed to create account"}, 500


@user_auth_bp.route('/login')
class LoginResource(Resource):
    @user_auth_bp.response(200, "Login successful")
    @user_auth_bp.response(400, "Validation error")
    @user_auth_bp.response(401, "Invalid credentials")
    @user_auth_bp.response(429, "Too many login attempts")
    @user_auth_bp.response(500, "Failed to authenticate")
    def post(self):
        """Verify credentials and set session cookie. Returns 401 on invalid credentials.

        After the password is verified against the account hash, the vault is opened
        with :meth:`~services.vault_service.VaultService.unlock_or_restore`, which:

        * ``"unlocked"``      — the live ``vault_config`` row opened normally.
        * ``"restored"``      — the live row was missing/corrupt but a filesystem
                                backup key matched; the vault was rebuilt and opened
                                with no data loss.
        * ``"unrecoverable"`` — neither the live row nor any backup opened. The DEK is
                                permanently lost, so the master account is wiped and a
                                401 with ``onboarding_required: True`` is returned to
                                force clean re-onboarding rather than logging the user
                                into an unusable vault.

        A transient error (DB locked, I/O) is caught and does NOT wipe the account —
        the login still succeeds with ``vault_state: "locked"``.

        On a successful open the post-unlock capability reconnection hook
        (:func:`_reconnect_capabilities`) is called so that capabilities that store
        encrypted credentials are re-connected immediately.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.auth_session_service import create_session
            from services.vault_service import (
                get_vault_service, OUTCOME_UNRECOVERABLE,
            )
            from flask import make_response

            data = request.get_json() or {}
            username = (data.get('username') or '').strip()
            password = data.get('password') or ''

            # Validation
            if not username or not password:
                return {"error": "Username and password required"}, 400

            if not _get_login_rate_limiter().is_allowed(request.remote_addr or "unknown"):
                return {"error": "Too many login attempts. Try again later."}, 429

            db = get_shared_db_service()

            # Fetch account and verify password hash
            with db.get_session() as session:
                row = session.execute(
                    text(
                        "SELECT password_hash FROM master_account "
                        "WHERE username = :username"
                    ),
                    {"username": username}
                ).fetchone()

                if not row or not check_password_hash(row[0], password):
                    return {"error": "Invalid credentials"}, 401

            # The password is now verified against the account hash. Open the vault
            # with it, recovering from a filesystem backup if the live vault_config
            # row is missing or corrupt. If neither the row nor any backup
            # opens, the DEK is permanently lost — wipe the account to force a clean
            # re-onboarding rather than logging the user into an unusable vault.
            vault = get_vault_service()
            vault_state = "locked"
            try:
                outcome = vault.unlock_or_restore(password)
            except Exception as exc:
                # Transient failure (DB locked, I/O error) — do NOT wipe the account.
                logger.error("[Auth] Vault open failed unexpectedly during login: %s", exc)
                resp = make_response(
                    jsonify({"ok": True, "vault_state": "locked"}),
                    200,
                )
                create_session(resp)
                return resp

            if outcome == OUTCOME_UNRECOVERABLE:
                logger.error(
                    "[Auth] Vault key corrupted with no valid backup — wiping the "
                    "master account to force re-onboarding."
                )
                with db.get_session() as session:
                    session.execute(text("DELETE FROM master_account"))
                    session.commit()
                vault.reset()
                return {
                    "error": "Your encryption key could not be recovered. "
                    "Please set up your account again.",
                    "onboarding_required": True,
                }, 401

            vault_state = "unlocked"
            _reconnect_capabilities()

            # Create session and set cookie
            resp = make_response(
                jsonify({"ok": True, "vault_state": vault_state}),
                200,
            )
            create_session(resp)
            return resp
        except Exception as e:
            logger.error(f"[REST API] Login error: {e}")
            return {"error": "Failed to authenticate"}, 500


@user_auth_bp.route('/logout')
class LogoutResource(Resource):
    @user_auth_bp.response(200, "Logged out")
    @user_auth_bp.response(500, "Failed to logout")
    def post(self):
        """Invalidate the current session and clear the cookie.

        Seals the vault by clearing the in-memory DEK before destroying the session
        so that no subsequent request can access encrypted data after the user has
        logged out.  Vault errors are caught and logged without preventing the
        session destruction.
        """
        try:
            from services.auth_session_service import destroy_session
            from services.vault_service import get_vault_service
            from flask import make_response

            # Seal the vault — wipe the DEK from memory
            try:
                get_vault_service().lock()
            except Exception as vault_exc:
                logger.warning("[Auth] Vault lock on logout failed: %s", vault_exc)

            resp = make_response(jsonify({"ok": True}), 200)
            destroy_session(request, resp)
            return resp
        except Exception as e:
            logger.error(f"[REST API] Logout error: {e}")
            return {"error": "Failed to logout"}, 500