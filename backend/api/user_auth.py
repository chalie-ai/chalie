"""
User authentication blueprint — /auth endpoints for master account.
"""

import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

user_auth_bp = Blueprint('user_auth', __name__)


@user_auth_bp.route('/auth/status', methods=['GET'])
def auth_status():
    """Check whether master account exists, providers are configured, and user has session."""
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
                text("SELECT COUNT(*) FROM providers WHERE is_active = TRUE")
            ).fetchone()[0]

        # Check session
        has_session = validate_session(request)

        return jsonify({
            "has_master_account": account_count > 0,
            "has_providers": provider_count > 0,
            "has_session": has_session,
        }), 200
    except Exception as e:
        logger.error(f"[REST API] Auth status error: {e}")
        return jsonify({"error": "Failed to check auth status"}), 500


@user_auth_bp.route('/auth/register', methods=['POST'])
def register():
    """Create master account. Fails (409) if one exists. Sets session cookie on success."""
    try:
        from services.database_service import get_shared_db_service
        from services.auth_session_service import create_session
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
                text("INSERT INTO master_account (username, password_hash) VALUES (:username, :password_hash)"),
                {"username": username, "password_hash": password_hash}
            )
            session.commit()

        # Create session and set cookie
        resp = make_response(jsonify({"ok": True}), 201)
        create_session(resp)
        return resp
    except Exception as e:
        logger.error(f"[REST API] Register error: {e}")
        return jsonify({"error": "Failed to create account"}), 500


@user_auth_bp.route('/auth/login', methods=['POST'])
def login():
    """Verify credentials and set session cookie. Returns 401 on invalid credentials."""
    try:
        from services.database_service import get_shared_db_service
        from services.auth_session_service import create_session
        from flask import make_response

        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''

        # Validation
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        db = get_shared_db_service()

        # Fetch account
        with db.get_session() as session:
            row = session.execute(
                text("SELECT password_hash FROM master_account WHERE username = :username"),
                {"username": username}
            ).fetchone()

            if not row or not check_password_hash(row[0], password):
                return jsonify({"error": "Invalid credentials"}), 401

        # Create session and set cookie
        resp = make_response(jsonify({"ok": True}), 200)
        create_session(resp)
        return resp
    except Exception as e:
        logger.error(f"[REST API] Login error: {e}")
        return jsonify({"error": "Failed to authenticate"}), 500


@user_auth_bp.route('/auth/logout', methods=['POST'])
def logout():
    """Invalidate the current session and clear the cookie."""
    try:
        from services.auth_session_service import destroy_session
        from flask import make_response

        resp = make_response(jsonify({"ok": True}), 200)
        destroy_session(request, resp)
        return resp
    except Exception as e:
        logger.error(f"[REST API] Logout error: {e}")
        return jsonify({"error": "Failed to logout"}), 500
