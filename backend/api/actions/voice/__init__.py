"""Voice action package — verb operations for readiness, playback, and transcription.

The three verbs mounted in ``api/routes.py`` reproduce the pre-migration URLs
exactly (``/api/voice/health``, ``/api/voice/transcript/<id>``,
``/api/voice/transcribe``), so moving the surface onto the Endpoint/Action
contract changed no client path.
"""
