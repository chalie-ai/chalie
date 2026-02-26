"""
Stub blueprint — future endpoints that return 501 Not Implemented.
"""

from flask import Blueprint, jsonify

from .auth import require_session

stubs_bp = Blueprint('stubs', __name__)

_NOT_IMPLEMENTED = ({"error": "Not implemented", "planned": True}, 501)


def _stub_route(rule, **options):
    """Register a stub route that returns 501 for all listed methods."""
    methods = options.pop('methods', ['GET'])

    def decorator(f):
        return stubs_bp.route(rule, methods=methods, **options)(f)
    return decorator


@stubs_bp.route('/calendar', methods=['GET'])
@stubs_bp.route('/calendar/<path:subpath>', methods=['GET'])
@require_session
def calendar_stub(subpath=None):
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/notifications/digest', methods=['GET'])
@require_session
def notifications_digest_stub():
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/integrations/messages', methods=['GET'])
@stubs_bp.route('/integrations/messages/<path:subpath>', methods=['GET'])
@require_session
def integrations_messages_stub(subpath=None):
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/integrations/messages/reply', methods=['POST'])
@require_session
def integrations_reply_stub():
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/permissions', methods=['GET'])
@require_session
def permissions_stub():
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]
