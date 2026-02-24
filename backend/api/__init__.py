"""
REST API package — Flask app factory with Blueprint registration.
"""

import os
import logging
from flask import Flask
from flask_cors import CORS

from .auth import require_session


logger = logging.getLogger(__name__)


def create_app():
    """Create and configure Flask application with all blueprints."""
    app = Flask(__name__)

    # Set secret key for cookie signing
    app.secret_key = os.environ.get('SESSION_SECRET_KEY', 'dev-secret-change-in-production')

    # CORS — allow all origins (single-user personal assistant)
    CORS(app)

    # Register blueprints
    from .user_auth import user_auth_bp
    from .system import system_bp
    from .conversation import conversation_bp
    from .memory import memory_bp
    from .proactive import proactive_bp
    from .privacy import privacy_bp
    from .stubs import stubs_bp
    from .push import push_bp
    from .tools import tools_bp
    from .providers import providers_bp
    from .scheduler import scheduler_bp
    from .lists import lists_bp
    from .moments import moments_bp

    app.register_blueprint(user_auth_bp)
    app.register_blueprint(system_bp)
    app.register_blueprint(conversation_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(proactive_bp)
    app.register_blueprint(privacy_bp)
    app.register_blueprint(stubs_bp)
    app.register_blueprint(push_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(providers_bp)
    app.register_blueprint(scheduler_bp)
    app.register_blueprint(lists_bp)
    app.register_blueprint(moments_bp)

    logger.info("[REST API] All blueprints registered")
    return app
