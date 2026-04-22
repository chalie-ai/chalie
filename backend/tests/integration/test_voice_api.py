"""
Integration: voice API — synthesize (single WAV blob) + transcribe + health.

Tests exercise the real Kokoro + Moonshine models.
Each test skips with pytest.skip() if /voice/health reports models unavailable,
so the suite stays green in environments without voice dependencies installed.

Uses the authed_client fixture from conftest.py (real Flask app, auth bypassed,
real SQLite + MemoryStore).
"""

import io
import pytest


def _voice_available(client) -> bool:
    """Return True only if /voice/health reports status='ok'."""
    resp = client.get('/voice/health')
    if resp.status_code != 200:
        return False
    data = resp.get_json()
    return (data or {}).get('status') == 'ok'


@pytest.mark.integration
def test_health_endpoint_shape(authed_client):
    """GET /voice/health → JSON with 'status' key holding a known string."""
    client, _db, _store = authed_client

    resp = client.get('/voice/health')
    # 200 when voice is available (ok or loading); 503 when unavailable
    assert resp.status_code in (200, 503)

    data = resp.get_json()
    assert data is not None
    assert 'status' in data
    assert data['status'] in ('ok', 'loading', 'unavailable')


@pytest.mark.integration
def test_synthesize_returns_single_wav_blob(authed_client):
    """POST /voice/synthesize with short text → 200, audio/wav, valid RIFF header, duration > 0.1s."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': 'Hello world.'},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert 'audio/wav' in resp.content_type

    body = resp.data
    # Valid WAV: RIFF header
    assert len(body) > 44, 'Response too short to be a WAV file'
    assert body[:4] == b'RIFF', 'Missing RIFF header'
    assert body[8:12] == b'WAVE', 'Missing WAVE marker'

    # Duration check via soundfile
    try:
        import soundfile as sf
        data, samplerate = sf.read(io.BytesIO(body))
        duration = len(data) / samplerate
        assert duration > 0.1, f'Audio too short: {duration:.3f}s'
    except ImportError:
        pass  # soundfile not installed — skip duration check


@pytest.mark.integration
@pytest.mark.parametrize(
    'text',
    ['', '   ', '\n\t  \n'],
    ids=['empty', 'whitespace', 'newlines-tabs'],
)
def test_synthesize_rejects_empty_text(authed_client, text):
    """POST /voice/synthesize → 400 for empty or whitespace-only input."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': text},
        content_type='application/json',
    )
    assert resp.status_code == 400

    data = resp.get_json()
    assert data is not None
    assert 'error' in data


@pytest.mark.integration
def test_synthesize_long_text_single_blob(authed_client):
    """POST /voice/synthesize with multi-sentence text → single blob, duration > 3s, valid WAV."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    long_text = (
        'The quick brown fox jumps over the lazy dog. '
        'She sells sea shells by the sea shore. '
        'How much wood would a woodchuck chuck if a woodchuck could chuck wood. '
        'Peter Piper picked a peck of pickled peppers. '
        'All that glitters is not gold, and not all who wander are lost.'
    )

    resp = client.post(
        '/voice/synthesize',
        json={'text': long_text},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert 'audio/wav' in resp.content_type

    body = resp.data
    assert body[:4] == b'RIFF', 'Missing RIFF header'
    assert body[8:12] == b'WAVE', 'Missing WAVE marker'

    try:
        import soundfile as sf
        data, samplerate = sf.read(io.BytesIO(body))
        duration = len(data) / samplerate
        assert duration > 3.0, f'Expected > 3s for 5 sentences, got {duration:.3f}s'
    except ImportError:
        pass  # soundfile not installed — skip duration check


@pytest.mark.integration
def test_transcribe_roundtrip(authed_client):
    """Synthesize 'hello chalie' → upload WAV → assert 'hello' in transcript (case-insensitive)."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    # Step 1: synthesize
    synth_resp = client.post(
        '/voice/synthesize',
        json={'text': 'hello chalie'},
        content_type='application/json',
    )
    assert synth_resp.status_code == 200
    wav_bytes = synth_resp.data

    # Step 2: transcribe the synthesized WAV
    transcribe_resp = client.post(
        '/voice/transcribe',
        data={'file': (io.BytesIO(wav_bytes), 'test.wav')},
        content_type='multipart/form-data',
    )
    assert transcribe_resp.status_code == 200

    data = transcribe_resp.get_json()
    assert data is not None
    assert 'text' in data

    text = (data.get('text') or '').lower()
    assert 'hello' in text, f"Expected 'hello' in transcript, got: {data.get('text')!r}"
