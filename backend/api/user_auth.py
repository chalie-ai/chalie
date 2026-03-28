"""
User authentication blueprint — /auth endpoints for master account.
"""

import logging
from flask import Blueprint, request, jsonify
from services.database_service import text
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

user_auth_bp = Blueprint('user_auth', __name__)


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
        from services.tool_library_service import register_tool

        capabilities = load_capabilities()
        reconnected = 0
        for cap in capabilities.values():
            if cap.connect():
                for tool in cap.get_tools():
                    register_tool(
                        tool['name'],
                        tool['handler'],
                        {k: v for k, v in tool.items() if k != 'handler'},
                    )
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
    except Exception:
        return "uninitialized"


@user_auth_bp.route('/auth/status', methods=['GET'])
def auth_status():
    """Check whether master account exists, providers are configured, and
    user has session.

    Returns a JSON body with the following keys:

    * ``has_master_account`` — ``True`` if at least one row exists in
      ``master_account``.
    * ``has_providers``      — ``True`` if at least one active provider is configured.
    * ``has_session``        — ``True`` if the request carries a valid session token.
    * ``vault_state``        — ``"unlocked" | "locked" | "uninitialized"``.
    """
    try:
        from services.database_service import get_shared_db_service
        from services.auth_session_service import validate_session

        db = get_shared_db_service()

        # Check master account
        with db.get_session() as session:
            account_count = session.execute(
                text("SELECT COUNT(*) FROM master_account")
            ).fetchone()[0]

        # Check providers (count only — avoids decryption which can fail if key changed)
        with db.get_session() as session:
            provider_count = session.execute(
                text("SELECT COUNT(*) FROM providers WHERE is_active = 1")
            ).fetchone()[0]

        # Check session
        has_session = validate_session(request)

        return jsonify({
            "has_master_account": account_count > 0,
            "has_providers": provider_count > 0,
            "has_session": has_session,
            "vault_state": _get_vault_state(),
        }), 200
    except Exception as e:
        logger.error(f"[REST API] Auth status error: {e}")
        return jsonify({"error": "Failed to check auth status"}), 500


@user_auth_bp.route('/auth/register', methods=['POST'])
def register():
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
            return jsonify({"error": "Username required"}), 400
        if len(password) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400

        db = get_shared_db_service()

        # Check if master account already exists
        with db.get_session() as session:
            existing = session.execute(
                text("SELECT COUNT(*) FROM master_account")
            ).fetchone()[0]

            if existing > 0:
                return jsonify({"error": "Master account already exists"}), 409

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
            return jsonify({
                "error": "Account created but vault "
                "initialization failed"
            }), 500

        # Create session and set cookie
        resp = make_response(jsonify({"ok": True, "vault_state": "unlocked"}), 201)
        create_session(resp)
        return resp
    except Exception as e:
        logger.error(f"[REST API] Register error: {e}")
        return jsonify({"error": "Failed to create account"}), 500


@user_auth_bp.route('/auth/login', methods=['POST'])
def login():
    """Verify credentials and set session cookie. Returns 401 on invalid credentials.

    After a successful password check the vault is unlocked with the same
    password.  If the vault returns ``False`` from
    :meth:`~services.vault_service.VaultService.unlock`
    (wrong password or vault inconsistency) the endpoint returns 401.  If the
    vault has never been initialised (``RuntimeError``) the login still succeeds
    but ``vault_state`` is ``"locked"`` in the response and a warning is logged.

    On a successful vault unlock the post-unlock capability reconnection hook
    (:func:`_reconnect_capabilities`) is called so that capabilities that store
    encrypted credentials are re-connected immediately.
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
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

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
                return jsonify({"error": "Invalid credentials"}), 401

        # Unlock the vault so encrypted credentials are accessible.
        # Existing users upgrading from pre-vault versions won't have a
        # vault_config row yet — initialize it on first login using the
        # password we just verified against the hash.
        vault = get_vault_service()
        vault_state = "locked"
        try:
            if vault.get_state() == "uninitialized":
                logger.info(
                    "[Auth] Vault uninitialized — "
                    "auto-initializing for existing user"
                )
                vault.initialize(password)
            unlocked = vault.unlock(password)
            if not unlocked:
                logger.warning(
                    "[Auth] vault.unlock() returned False "
                    "despite correct password"
                )
                return jsonify({"error": "Invalid credentials"}), 401
            vault_state = "unlocked"
            _reconnect_capabilities()
        except Exception as exc:
            logger.error("[Auth] Vault unlock failed during login: %s", exc)

        # Create session and set cookie
        resp = make_response(jsonify({"ok": True, "vault_state": vault_state}), 200)
        create_session(resp)
        return resp
    except Exception as e:
        logger.error(f"[REST API] Login error: {e}")
        return jsonify({"error": "Failed to authenticate"}), 500


@user_auth_bp.route('/auth/logout', methods=['POST'])
def logout():
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
        return jsonify({"error": "Failed to logout"}), 500
