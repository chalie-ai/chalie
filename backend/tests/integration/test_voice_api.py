"""
Integration: voice API — synthesize (single WAV blob) + transcribe + health.

Tests exercise the real kokoro-onnx + moonshine-onnx models via the real Flask
app. Each test that requires model files skips cleanly when the model files are
absent (resources/voice-models/), so the suite stays green in CI environments
without voice dependencies installed.

The pure-function tests in test_voice_xml_strip.py run unconditionally; these
integration tests are for the full HTTP contract only.

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


def _wav_duration(data: bytes) -> float:
    """Parse WAV header for duration without decoding. Returns 0.0 on bad input."""
    import struct
    try:
        if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            return 0.0
        channels = struct.unpack_from('<H', data, 22)[0]
        sample_rate = struct.unpack_from('<I', data, 24)[0]
        bits_per_sample = struct.unpack_from('<H', data, 34)[0]
        if sample_rate == 0 or channels == 0 or bits_per_sample == 0:
            return 0.0
        offset = 12
        while offset < len(data) - 8:
            chunk_id = data[offset:offset + 4]
            chunk_size = struct.unpack_from('<I', data, offset + 4)[0]
            if chunk_id == b'data':
                return chunk_size / (sample_rate * channels * (bits_per_sample // 8))
            offset += 8 + chunk_size
        return 0.0
    except Exception:
        return 0.0


@pytest.mark.integration
def test_health_endpoint_returns_known_status(authed_client):
    """GET /voice/health → 200 with status in {ok, loading, unavailable}."""
    client, _db, _store = authed_client

    resp = client.get('/voice/health')
    assert resp.status_code == 200

    data = resp.get_json()
    assert data is not None
    assert 'status' in data
    assert data['status'] in ('ok', 'loading', 'unavailable')


@pytest.mark.integration
def test_synthesize_returns_single_wav_blob(authed_client):
    """POST /voice/synthesize → 200 audio/wav blob with non-zero duration."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': 'Hello world.'},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'

    body = resp.get_data()
    assert body[:4] == b'RIFF', 'Response body must start with RIFF WAV header'
    assert body[8:12] == b'WAVE', 'Response body must contain WAVE marker'
    assert len(body) > 44, 'WAV blob too short to contain any audio'

    duration = _wav_duration(body)
    assert duration > 0.1, 'Synthesized audio must have non-zero duration (got %fs)' % duration


@pytest.mark.integration
def test_synthesize_rejects_empty_text(authed_client):
    """POST /voice/synthesize → 400 for empty or whitespace-only input."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    for text in ('', '   ', '\n\t  \n'):
        resp = client.post(
            '/voice/synthesize',
            json={'text': text},
            content_type='application/json',
        )
        assert resp.status_code == 400, 'Expected 400 for input %r, got %d' % (text, resp.status_code)
        data = resp.get_json()
        assert data is not None
        assert 'error' in data
        assert data['error'] == 'Text is required'


@pytest.mark.integration
def test_synthesize_markdown_input_produces_audio(authed_client):
    """Markdown-rich LLM response (bold, italic, code span, list, link) → non-trivial WAV.

    Regression guard for the _clean_for_tts pipeline: if markdown markers
    survive into the phonemizer as literal characters, Kokoro either skips them
    silently or produces very short audio. Duration > 0.5s is the observable
    proxy that real speech was synthesized.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    markdown_text = (
        "Here are **three key points** about Python:\n\n"
        "- Use `print()` for output\n"
        "- Prefer *explicit* over implicit\n"
        "- See [the docs](https://docs.python.org) for more\n\n"
        "The _language_ was designed for readability."
    )

    resp = client.post(
        '/voice/synthesize',
        json={'text': markdown_text},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'
    body = resp.get_data()
    assert _wav_duration(body) > 0.5, 'Markdown-rich text must produce substantial audio'


@pytest.mark.integration
def test_synthesize_html_input_produces_audio(authed_client):
    """HTML input (<p>, <strong>, <em>) strips cleanly and synthesizes to audio.

    If the HTML pre-pass in _clean_for_tts is removed or broken, literal tag
    text reaches the phonemizer and produces garbled or empty output.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    html_text = (
        '<p>The capital of France is <strong>Paris</strong>.</p>'
        '<p>It sits on the <em>Seine</em> river.</p>'
    )

    resp = client.post(
        '/voice/synthesize',
        json={'text': html_text},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'
    body = resp.get_data()
    assert _wav_duration(body) > 0.1, 'HTML input must produce audible audio'


@pytest.mark.integration
def test_synthesize_url_input_produces_audio(authed_client):
    """Input containing a URL → audio (spoken-host form, not raw URL characters).

    The observable invariant is that audio is produced at all — espeak-ng would
    struggle or stutter reading raw slash-and-digit sequences, but the
    spoken-host form ('google dot com') is a normal English phrase.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': 'Read more at http://google.com/search for details.'},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'
    body = resp.get_data()
    assert _wav_duration(body) > 0.1, 'URL-containing input must produce audio'


@pytest.mark.integration
def test_transcribe_empty_file_returns_400(authed_client):
    """POST /voice/transcribe with an empty file body → 400."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/transcribe',
        data={'file': (io.BytesIO(b''), 'empty.wav')},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert 'error' in data


@pytest.mark.integration
def test_transcribe_missing_file_field_returns_400(authed_client):
    """POST /voice/transcribe with no file field → 400."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/transcribe',
        data={},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert 'error' in data
