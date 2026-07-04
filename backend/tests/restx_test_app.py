"""Mount flask-restx Namespace(s) on a throwaway Flask app for unit tests.

Mirrors the production ``create_app()`` registration (a fresh ``Api`` per app,
Swagger UI disabled, deferred ``init_app``) so a test can exercise a single
namespace in isolation. Replaces the pre-migration ``app.register_blueprint(ns)``
pattern, which no longer works now that the API surface is flask-restx
Namespaces (a ``Namespace`` has no ``register`` method).
"""

from flask import Flask
from flask_restx import Api


def mount_namespace(*namespaces: object) -> Flask:
    """Build a Flask app with the given Namespace(s) registered, and return it."""
    app = Flask(__name__)
    api = Api(doc=False)
    for ns in namespaces:
        api.add_namespace(ns)
    api.init_app(app)
    return app
