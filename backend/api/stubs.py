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
    """Return 501 Not Implemented for all calendar endpoints.

    Placeholder for future calendar integration.  Accepts an optional
    ``subpath`` so that any nested calendar URL is handled gracefully
    rather than falling through to a 404.

    Args:
        subpath: Optional URL sub-path captured from
            ``/calendar/<path:subpath>``.  Ignored at this time.

    Returns:
        A JSON response with ``{"error": "Not implemented", "planned": true}``
        and HTTP status 501.
    """
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/notifications/digest', methods=['GET'])
@require_session
def notifications_digest_stub():
    """Return 501 Not Implemented for the notifications digest endpoint.

    Placeholder for a future aggregated notifications digest feed.

    Returns:
        A JSON response with ``{"error": "Not implemented", "planned": true}``
        and HTTP status 501.
    """
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/integrations/messages', methods=['GET'])
@stubs_bp.route('/integrations/messages/<path:subpath>', methods=['GET'])
@require_session
def integrations_messages_stub(subpath=None):
    """Return 501 Not Implemented for all integrations message-listing endpoints.

    Placeholder for future third-party messaging integration.  Accepts an
    optional ``subpath`` so nested URLs are handled without 404s.

    Args:
        subpath: Optional URL sub-path captured from
            ``/integrations/messages/<path:subpath>``.  Ignored at this time.

    Returns:
        A JSON response with ``{"error": "Not implemented", "planned": true}``
        and HTTP status 501.
    """
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/integrations/messages/reply', methods=['POST'])
@require_session
def integrations_reply_stub():
    """Return 501 Not Implemented for the integrations message-reply endpoint.

    Placeholder for a future API to send replies via integrated messaging
    platforms.

    Returns:
        A JSON response with ``{"error": "Not implemented", "planned": true}``
        and HTTP status 501.
    """
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]


@stubs_bp.route('/permissions', methods=['GET'])
@require_session
def permissions_stub():
    """Return 501 Not Implemented for the permissions endpoint.

    Placeholder for a future fine-grained permissions / capability query API.

    Returns:
        A JSON response with ``{"error": "Not implemented", "planned": true}``
        and HTTP status 501.
    """
    return jsonify(_NOT_IMPLEMENTED[0]), _NOT_IMPLEMENTED[1]
