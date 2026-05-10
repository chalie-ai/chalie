"""
Integration: voice API — synthesize (NDJSON streaming) + transcribe + health.

Tests exercise the real Kokoro + Moonshine models. Each test skips with
pytest.skip() if /voice/health reports models unavailable, so the suite
stays green in environments without voice dependencies installed.

Uses the authed_client fixture from conftest.py (real Flask app, auth
bypassed, real SQLite + MemoryStore).
"""

import base64
import io
import json

import pytest


def _voice_available(client) -> bool:
    """Return True only if /voice/health reports status='ok'."""
    resp = client.get('/voice/health')
    if resp.status_code != 200:
        return False
    data = resp.get_json()
    return (data or {}).get('status') == 'ok'


def _read_ndjson(resp):
    """Decode an NDJSON streaming response into a list of payloads."""
    body = resp.get_data(as_text=True)
    chunks = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks.append(json.loads(line))
    return chunks


def _decode_chunk_wav(chunk: dict) -> bytes:
    """Base64-decode the ``audio`` field of an NDJSON chunk."""
    return base64.b64decode(chunk['audio'])


@pytest.mark.integration
def test_health_endpoint_shape(authed_client):
    """GET /voice/health → JSON with 'status' key holding a known string."""
    client, _db, _store = authed_client

    resp = client.get('/voice/health')
    assert resp.status_code == 200

    data = resp.get_json()
    assert data is not None
    assert 'status' in data
    assert data['status'] in ('ok', 'loading', 'unavailable')


@pytest.mark.integration
def test_synthesize_streams_ndjson_chunks(authed_client):
    """POST /voice/synthesize streams NDJSON chunks; each is a valid WAV."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': 'Hello world.'},
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.mimetype == 'application/x-ndjson'

    payloads = _read_ndjson(resp)
    assert len(payloads) >= 2, 'Expected at least one chunk + a done sentinel'

    chunks = [p for p in payloads if not p.get('done')]
    sentinels = [p for p in payloads if p.get('done')]
    assert len(sentinels) == 1, 'Expected exactly one done sentinel'
    assert len(chunks) >= 1, 'Expected at least one audio chunk'

    total = chunks[0]['total']
    assert len(chunks) == total, f'Got {len(chunks)} chunks, expected total={total}'
    assert sentinels[0]['total'] == total

    assert [c['index'] for c in chunks] == list(range(total))

    wav_bytes = _decode_chunk_wav(chunks[0])
    assert len(wav_bytes) > 44, 'Chunk too short to be a WAV file'
    assert wav_bytes[:4] == b'RIFF', 'Missing RIFF header'
    assert wav_bytes[8:12] == b'WAVE', 'Missing WAVE marker'

    try:
        import soundfile as sf
        data, samplerate = sf.read(io.BytesIO(wav_bytes))
        duration = len(data) / samplerate
        assert duration > 0.1, f'Audio too short: {duration:.3f}s'
    except ImportError:
        pass


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
def test_synthesize_long_text_returns_full_audio(authed_client):
    """Long multi-sentence text returns multiple WAV chunks covering the whole message.

    With the streaming pipeline each sentence is yielded as a separate chunk,
    so len(chunks) >= 1 (one per sentence). The combined WAV duration across
    all chunks must still cover substantial audio.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    sentence = (
        'The quick brown fox jumps over the lazy dog. '
        'She sells sea shells by the sea shore. '
        'How much wood would a woodchuck chuck if a woodchuck could chuck wood. '
        'Peter Piper picked a peck of pickled peppers. '
        'All that glitters is not gold and not all who wander are lost. '
    )
    long_text = sentence * 4  # ~1150 chars, ~20 sentences

    resp = client.post(
        '/voice/synthesize',
        json={'text': long_text},
        content_type='application/json',
    )

    assert resp.status_code == 200
    payloads = _read_ndjson(resp)
    chunks = [p for p in payloads if not p.get('done')]
    sentinels = [p for p in payloads if p.get('done')]

    assert len(chunks) >= 1, f'Expected at least one audio chunk, got {len(chunks)}'
    assert len(sentinels) == 1, 'Expected exactly one done sentinel'
    assert sentinels[0]['total'] == len(chunks)

    wav = _decode_chunk_wav(chunks[0])
    assert wav[:4] == b'RIFF'
    assert wav[8:12] == b'WAVE'

    try:
        import soundfile as sf
        total_duration = 0.0
        for chunk in chunks:
            data, samplerate = sf.read(io.BytesIO(_decode_chunk_wav(chunk)))
            total_duration += len(data) / samplerate
        assert total_duration > 3.0, f'Expected > 3s total audio, got {total_duration:.3f}s'
    except ImportError:
        pass


@pytest.mark.integration
def test_synthesize_streams_per_sentence(authed_client):
    """Multi-sentence input yields one audio chunk per sentence before the done sentinel.

    The streaming pipeline segments text with pysbd and yields one WAV chunk
    per sentence. Three input sentences must produce at least 3 chunks. The
    done sentinel must be the final NDJSON line and its total must equal the
    number of audio chunks emitted.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    resp = client.post(
        '/voice/synthesize',
        json={'text': 'Sentence one. Sentence two. Sentence three.'},
        content_type='application/json',
    )

    assert resp.status_code == 200
    payloads = _read_ndjson(resp)

    chunks = [p for p in payloads if not p.get('done')]
    sentinels = [p for p in payloads if p.get('done')]

    assert len(chunks) >= 3, f'Expected >= 3 audio chunks, got {len(chunks)}'
    assert len(sentinels) == 1, 'Expected exactly one done sentinel'
    assert sentinels[0]['total'] == len(chunks), (
        f"Sentinel total={sentinels[0]['total']} does not match chunk count={len(chunks)}"
    )

    # Sentinel must be the final NDJSON line
    assert payloads[-1].get('done'), 'done sentinel must be the last NDJSON line'

    # Chunk indices must be a contiguous 0-based sequence
    assert [c['index'] for c in chunks] == list(range(len(chunks)))


@pytest.mark.integration
def test_transcribe_roundtrip(authed_client):
    """Synthesize 'hello chalie' → feed the chunk WAV back to /voice/transcribe."""
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    synth_resp = client.post(
        '/voice/synthesize',
        json={'text': 'hello chalie'},
        content_type='application/json',
    )
    assert synth_resp.status_code == 200

    payloads = _read_ndjson(synth_resp)
    chunks = [p for p in payloads if not p.get('done')]
    assert len(chunks) >= 1

    wav_bytes = _decode_chunk_wav(chunks[0])

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
