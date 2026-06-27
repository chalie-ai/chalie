"""
Stub namespace — future endpoints that return 501 Not Implemented.
"""

from flask_restx import Namespace, Resource

from .auth import require_session

stubs_bp = Namespace('stubs', description='Future endpoints (not implemented)')

_NOT_IMPLEMENTED = ({"error": "Not implemented", "planned": True}, 501)


@stubs_bp.route('/calendar')
@stubs_bp.route('/calendar/<path:subpath>')
class CalendarStubResource(Resource):
    @require_session
    @stubs_bp.response(501, "Not implemented")
    def get(self, subpath: str | None = None):
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
        return _NOT_IMPLEMENTED[0], _NOT_IMPLEMENTED[1]


@stubs_bp.route('/notifications/digest')
class NotificationsDigestStubResource(Resource):
    @require_session
    @stubs_bp.response(501, "Not implemented")
    def get(self):
        """Return 501 Not Implemented for the notifications digest endpoint.

        Placeholder for a future aggregated notifications digest feed.

        Returns:
            A JSON response with ``{"error": "Not implemented", "planned": true}``
            and HTTP status 501.
        """
        return _NOT_IMPLEMENTED[0], _NOT_IMPLEMENTED[1]


@stubs_bp.route('/integrations/messages')
@stubs_bp.route('/integrations/messages/<path:subpath>')
class IntegrationsMessagesStubResource(Resource):
    @require_session
    @stubs_bp.response(501, "Not implemented")
    def get(self, subpath: str | None = None):
        """Return 501 Not Implemented for all integrations message-listing endpoints.

        Placeholder for future third-party messaging integration.  Accepts an
        optional ``_subpath`` so nested URLs are handled without 404s.

        Args:
            _subpath: Optional URL sub-path captured from
                ``/integrations/messages/<path:subpath>``.  Ignored at this time.

        Returns:
            A JSON response with ``{"error": "Not implemented", "planned": true}``
            and HTTP status 501.
        """
        return _NOT_IMPLEMENTED[0], _NOT_IMPLEMENTED[1]


@stubs_bp.route('/integrations/messages/reply')
class IntegrationsReplyStubResource(Resource):
    @require_session
    @stubs_bp.response(501, "Not implemented")
    def post(self):
        """Return 501 Not Implemented for the integrations message-reply endpoint.

        Placeholder for a future API to send replies via integrated messaging
        platforms.

        Returns:
            A JSON response with ``{"error": "Not implemented", "planned": true}``
            and HTTP status 501.
        """
        return _NOT_IMPLEMENTED[0], _NOT_IMPLEMENTED[1]


@stubs_bp.route('/permissions')
class PermissionsStubResource(Resource):
    @require_session
    @stubs_bp.response(501, "Not implemented")
    def get(self):
        """Return 501 Not Implemented for the permissions endpoint.

        Placeholder for a future fine-grained permissions / capability query API.

        Returns:
            A JSON response with ``{"error": "Not implemented", "planned": true}``
            and HTTP status 501.
        """
        return _NOT_IMPLEMENTED[0], _NOT_IMPLEMENTED[1]