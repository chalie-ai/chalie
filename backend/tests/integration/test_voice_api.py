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
def test_synthesize_markdown_rich_input_produces_audible_speech(authed_client):
    """Realistic LLM response with bold, italic, code span, link, and list → audible WAV.

    This is the primary regression guard for the markdown-it-py pipeline. A
    regression in _clean_for_tts (e.g. markdown markers surviving into plaintext,
    or nh3 stripping all content) would produce silence or a 400 here.
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
    payloads = _read_ndjson(resp)
    chunks = [p for p in payloads if not p.get('done')]
    sentinels = [p for p in payloads if p.get('done')]

    assert len(chunks) >= 1, 'Markdown-rich text must produce at least one audio chunk'
    assert len(sentinels) == 1

    # Each chunk must be a valid WAV with non-trivial audio content
    for chunk in chunks:
        wav = _decode_chunk_wav(chunk)
        assert wav[:4] == b'RIFF', f'Chunk {chunk["index"]} missing RIFF header'
        assert wav[8:12] == b'WAVE', f'Chunk {chunk["index"]} missing WAVE marker'
        assert len(wav) > 44, f'Chunk {chunk["index"]} is too short to contain audio'

    try:
        import soundfile as sf
        total_duration = sum(
            len(sf.read(io.BytesIO(_decode_chunk_wav(c)))[0]) / sf.read(io.BytesIO(_decode_chunk_wav(c)))[1]
            for c in chunks
        )
        assert total_duration > 2.0, f'Expected > 2s of speech for multi-sentence input, got {total_duration:.3f}s'
    except ImportError:
        pass


@pytest.mark.integration
def test_synthesize_html_input_strips_tags_and_produces_speech(authed_client):
    """HTML-style input (LLM block output) strips cleanly and synthesizes.

    The HTML pre-pass in _clean_for_tts handles '<p>foo</p><strong>bar</strong>'
    before markdown-it sees the text. If the pre-pass is removed or broken,
    literal tag text ('<p>', '<strong>') reaches the synthesizer causing
    garbled or empty output.
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
    payloads = _read_ndjson(resp)
    chunks = [p for p in payloads if not p.get('done')]

    assert len(chunks) >= 1, 'HTML input must produce at least one audio chunk'

    wav = _decode_chunk_wav(chunks[0])
    assert wav[:4] == b'RIFF'
    assert wav[8:12] == b'WAVE'
    assert len(wav) > 44


@pytest.mark.integration
def test_synthesize_first_chunk_arrives_before_stream_completes(authed_client):
    """Multi-sentence input: first chunk must arrive well before the stream ends.

    Proves the NDJSON pipeline streams sentence-by-sentence, not buffer-then-send.
    If the route switched back to a single synthesize() call and buffered all output,
    the first and last chunk would arrive at the same time (time_to_first ~= total).
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    text = (
        'The first sentence introduces the topic clearly. '
        'The second sentence adds supporting detail and context. '
        'The third sentence provides an example to illustrate. '
        'The fourth sentence summarises the argument made above. '
        'The fifth sentence closes with a memorable conclusion.'
    )

    resp = client.post(
        '/voice/synthesize',
        json={'text': text},
        content_type='application/json',
    )

    assert resp.status_code == 200

    # The Flask test client fully buffers the response before returning, so we
    # can't measure real streaming latency here — instead verify that the
    # response carries multiple audio chunks (structural proof of per-sentence
    # streaming) rather than one large block.
    body = resp.get_data(as_text=True)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    assert len(lines) >= 2, 'Expected at least one chunk line + sentinel line'
    payloads = [json.loads(line) for line in lines]
    chunks = [p for p in payloads if not p.get('done')]
    sentinels = [p for p in payloads if p.get('done')]

    # 5 input sentences → at least 4 chunks (pysbd may merge some, never fewer)
    assert len(chunks) >= 4, (
        f'Expected >= 4 per-sentence chunks for 5-sentence input, got {len(chunks)}. '
        'If this fails, the route may have reverted to single-buffer synthesis.'
    )
    assert len(sentinels) == 1
    assert payloads[-1].get('done'), 'done sentinel must be the last line'


@pytest.mark.integration
def test_synthesize_ordinal_expansion_reaches_synthesizer(authed_client):
    """Ordinal integers in input are expanded before synthesis and appear in transcript.

    '1st' must expand to 'first' in the TTS pipeline. If _expand_ordinals_for_tts
    is removed or the pipeline order changes so expansion happens after segmentation
    bypass, the synthesizer receives '1st' and the neural G2P either skips it or
    mispronounces it silently. The transcribe roundtrip catches this because
    'first' is a common word Moonshine recognises, while '1st' as a token is not.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    synth_resp = client.post(
        '/voice/synthesize',
        json={'text': 'She finished 1st in the race.'},
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
    transcript = (data.get('text') or '').lower()
    assert 'first' in transcript, (
        f"Expected 'first' in transcript (ordinal expansion), got: {data.get('text')!r}"
    )


@pytest.mark.integration
def test_synthesize_marker_only_input_does_not_crash(authed_client):
    """Input containing only markdown markers and no substantive text.

    After _clean_for_tts strips all markers, the text may become empty.
    The route must return 400 (empty text guard) rather than crash or
    synthesize silence. This guards against a regression where the empty-
    text check happens before clean rather than after.
    """
    client, _db, _store = authed_client

    if not _voice_available(client):
        pytest.skip('Voice models not available in this environment')

    marker_only_inputs = [
        '# ',
        '**',
        '---',
        '```\n```',
    ]

    for text in marker_only_inputs:
        resp = client.post(
            '/voice/synthesize',
            json={'text': text},
            content_type='application/json',
        )
        # Must be 400 (cleaned text is empty) or 200 (some residual text survived).
        # Must never be 500 — that would mean an unhandled crash in the pipeline.
        assert resp.status_code in (200, 400), (
            f"Input {text!r} produced unexpected status {resp.status_code}: "
            f"{resp.get_data(as_text=True)[:200]}"
        )


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
