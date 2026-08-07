"""User authentication namespace — /api/auth endpoints for master account."""

import logging
from typing import cast

from flask import request, jsonify
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource
from werkzeug.security import generate_password_hash

from services.auth_service import AuthService
from services.feature_flags import internal_dev_enabled
from contracts.constants.auth import LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW_SECONDS
from .auth import require_auth, _cookie_only, internal_only
from .dto import Error, expects, responds, register_dto
from .dto.auth import AuthStatus, LoginRequest, RegisterRequest, Username, VaultResult


logger = logging.getLogger(__name__)

user_auth_ns = Namespace('user_auth', description='Master account authentication', path='/api/auth')

register_dto(user_auth_ns, AuthStatus, Username, RegisterRequest, LoginRequest, VaultResult, Error)

_m = user_auth_ns.models


@user_auth_ns.route('/status')
class AuthStatusResource(Resource):
    @user_auth_ns.response(200, "Auth status", _m["AuthStatus"])
    @user_auth_ns.response(500, "Failed to check auth status", _m["Error"])
    @responds(AuthStatus)
    def get(self) -> AuthStatus | ResponseReturnValue:
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
            from services.database import Database
            from services.auth_session_service import AuthSessionService

            conn = Database.conn()
            account_count = cast("tuple[int, ...]", conn.execute(
                "SELECT COUNT(*) FROM master_account"
            ).fetchone())[0]
            provider_count = cast("tuple[int, ...]", conn.execute(
                "SELECT COUNT(*) FROM providers"
            ).fetchone())[0]

            has_session = AuthSessionService.validate_session(request)
            vault_state = AuthService().vault_state()
            if has_session and vault_state == "locked":
                has_session = False

            try:
                from services.provider_db_service import ProviderDbService
                has_vision = (
                    ProviderDbService().get_vision_provider()
                    is not None
                )
            except Exception:
                has_vision = False

            return AuthStatus(
                has_master_account=account_count > 0,
                has_providers=provider_count > 0,
                has_session=has_session,
                vault_state=vault_state,
                has_vision_provider=has_vision,
                internal_dev=internal_dev_enabled(),
            )
        except Exception as e:
            logger.error(f"[REST API] Auth status error: {e}")
            return {"error": "Failed to check auth status"}, 500


@user_auth_ns.route('/username')
class UsernameResource(Resource):
    @internal_only
    @require_auth
    @_cookie_only
    @user_auth_ns.doc(security="cookieAuth")
    @user_auth_ns.response(200, "Username", _m["Username"])
    @user_auth_ns.response(401, "Not authenticated")
    @user_auth_ns.response(403, "Internal only")
    @user_auth_ns.response(404, "No master account")
    @user_auth_ns.response(500, "Failed to read username", _m["Error"])
    @responds(Username)
    def get(self) -> Username | ResponseReturnValue:
        """Return the master account LOGIN username for the authenticated dashboard
        session — the credential the device's UnlockVault screen submits to
        POST /api/auth/login. Cookie-session only; a wrapper bearer must not read it.
        """
        try:
            from services.database import Database

            row = Database.conn().execute(
                "SELECT username FROM master_account LIMIT 1"
            ).fetchone()
            if not row:
                return {"error": "No master account"}, 404
            return Username(username=row[0])
        except Exception:
            logger.exception("[REST API] Get username error")
            return {"error": "Failed to read username"}, 500


@user_auth_ns.route('/register')
class RegisterResource(Resource):
    @user_auth_ns.expect(_m["RegisterRequest"])
    @user_auth_ns.response(201, "Account created", _m["VaultResult"])
    @user_auth_ns.response(409, "Master account already exists", _m["Error"])
    @user_auth_ns.response(422, "Validation failed", _m["Error"])
    @user_auth_ns.response(500, "Failed to create account", _m["Error"])
    @expects(RegisterRequest)
    def post(self, dto: RegisterRequest) -> ResponseReturnValue:
        """Create master account. Fails (409) if one exists. Sets session cookie on success.

        On success the vault is initialised with the master password and immediately
        unlocked so that the caller receives ``vault_state: "unlocked"`` in the
        response body.  If vault initialisation fails after the account row has been
        committed the endpoint returns 500 — the account exists but the vault must be
        re-initialised (this scenario is logged clearly for operator investigation).
        """
        try:
            from services.database import Database
            from services.auth_session_service import AuthSessionService
            from services.vault_service import get_vault_service
            from flask import make_response

            username = dto.username.strip()
            password = dto.password

            with Database.transaction() as conn:
                existing = cast("tuple[int, ...]", conn.execute(
                    "SELECT COUNT(*) FROM master_account"
                ).fetchone())[0]

                if existing > 0:
                    return {"error": "Master account already exists"}, 409

                password_hash = generate_password_hash(password)
                conn.execute(
                    "INSERT INTO master_account "
                    "(username, password_hash) "
                    "VALUES (:username, :password_hash)",
                    {"username": username, "password_hash": password_hash}
                )

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

            resp = make_response(jsonify({"ok": True, "vault_state": "unlocked"}), 201)
            AuthSessionService.create_session(resp)
            return resp
        except Exception as e:
            logger.error(f"[REST API] Register error: {e}")
            return {"error": "Failed to create account"}, 500


@user_auth_ns.route('/login')
class LoginResource(Resource):
    @user_auth_ns.expect(_m["LoginRequest"])
    @user_auth_ns.response(200, "Login successful", _m["VaultResult"])
    @user_auth_ns.response(401, "Invalid credentials", _m["Error"])
    @user_auth_ns.response(422, "Validation failed", _m["Error"])
    @user_auth_ns.response(429, "Too many login attempts", _m["Error"])
    @user_auth_ns.response(500, "Failed to authenticate", _m["Error"])
    @expects(LoginRequest)
    def post(self, dto: LoginRequest) -> ResponseReturnValue:
        """Verify credentials and set session cookie. Returns 401 on invalid credentials.

        The master password is verified against the account hash, the vault is opened
        with the supplied password, and a session cookie is issued on success. The
        resulting vault state is returned in the response body.
        """
        try:
            from services.auth_session_service import AuthSessionService
            from services.wrapper_rate_limiter import WrapperRateLimiter
            from flask import make_response

            username = dto.username.strip()
            password = dto.password

            if not WrapperRateLimiter(
                limit=LOGIN_RATE_LIMIT, window_seconds=LOGIN_RATE_WINDOW_SECONDS
            ).is_allowed(request.remote_addr or "unknown"):
                return {"error": "Too many login attempts. Try again later."}, 429

            vault_state = AuthService().login(username, password)
            if vault_state is None:
                return {"error": "Invalid credentials"}, 401

            resp = make_response(
                jsonify({"ok": True, "vault_state": vault_state}),
                200,
            )
            AuthSessionService.create_session(resp)
            return resp
        except Exception as e:
            logger.error(f"[REST API] Login error: {e}")
            return {"error": "Failed to authenticate"}, 500


@user_auth_ns.route('/logout')
class LogoutResource(Resource):
    @user_auth_ns.response(200, "Logged out", _m["VaultResult"])
    @user_auth_ns.response(500, "Failed to logout", _m["Error"])
    def post(self) -> ResponseReturnValue:
        """Invalidate the current session and clear the cookie.

        Seals the vault by clearing the in-memory DEK before destroying the session
        so that no subsequent request can access encrypted data after the user has
        logged out.  Vault errors are caught and logged without preventing the
        session destruction.
        """
        try:
            from services.auth_session_service import AuthSessionService
            from services.vault_service import get_vault_service
            from flask import make_response

            try:
                get_vault_service().lock()
            except Exception as vault_exc:
                logger.warning("[Auth] Vault lock on logout failed: %s", vault_exc)

            resp = make_response(jsonify({"ok": True}), 200)
            AuthSessionService.destroy_session(request, resp)
            return resp
        except Exception as e:
            logger.error(f"[REST API] Logout error: {e}")
            return {"error": "Failed to logout"}, 500
