"""Unit tests for Doc2QueryService — raw ONNX Runtime implementation.

Covers loading, generation, KV cache handling, top-p sampling,
idle unloading, and thread safety.  No real ONNX models are loaded;
all sessions and the tokenizer are mocked.
"""

import threading
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder_session(vocab_size=32128, hidden=512):
    """Return a mock encoder InferenceSession.

    run() returns a single encoder_hidden_states array of shape (1, seq, hidden).
    """
    session = MagicMock()
    session.get_outputs.return_value = [MagicMock(name="last_hidden_state")]
    session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]
    session.run.return_value = [np.zeros((1, 5, hidden), dtype=np.float32)]
    return session


class _NamedOutput:
    """Minimal stand-in for an ONNX session output descriptor with a real .name string."""
    def __init__(self, name: str):
        self.name = name


def _make_decoder_session(vocab_size=32128, num_layers=6):
    """Return a mock decoder InferenceSession (step 0).

    Outputs: logits (1, 1, vocab_size) + present KV tensors for each layer.
    """
    session = MagicMock()
    output_descriptors = [_NamedOutput("logits")] + [
        _NamedOutput(f"present.{i}.decoder.key") for i in range(num_layers)
    ]
    session.get_outputs.return_value = output_descriptors
    session.get_inputs.return_value = [
        _NamedOutput("input_ids"),
        _NamedOutput("encoder_attention_mask"),
        _NamedOutput("encoder_hidden_states"),
    ]
    # logits: shape (1, 1, vocab_size) — token 2 always wins after softmax
    logits = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
    logits[0, 0, 2] = 10.0  # token 2 dominates
    kv_arrays = [np.zeros((1, 8, 1, 64), dtype=np.float32) for _ in range(num_layers)]
    session.run.return_value = [logits] + kv_arrays
    return session


def _make_decoder_past_session(vocab_size=32128, num_layers=6):
    """Return a mock decoder_with_past InferenceSession (steps 1+).

    Outputs eos_token_id=1 on the first call, stopping generation early.
    """
    session = MagicMock()
    output_descriptors = [_NamedOutput("logits")] + [
        _NamedOutput(f"present.{i}.decoder.key") for i in range(num_layers)
    ]
    session.get_outputs.return_value = output_descriptors
    session.get_inputs.return_value = [
        _NamedOutput("input_ids"),
        _NamedOutput("encoder_attention_mask"),
    ] + [_NamedOutput(f"past_key_values.{i}.decoder.key") for i in range(num_layers)]

    # Return EOS (token 1) so generation stops after first step
    eos_logits = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
    eos_logits[0, 0, 1] = 10.0
    kv_arrays = [np.zeros((1, 8, 1, 64), dtype=np.float32) for _ in range(num_layers)]
    session.run.return_value = [eos_logits] + kv_arrays
    return session


def _make_tokenizer(decode_text="what is this about"):
    """Return a mock AutoTokenizer."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
        "attention_mask": np.ones((1, 5), dtype=np.int64),
    }
    tok.decode.return_value = decode_text
    return tok


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

class TestDoc2QueryLoading:

    def test_is_available_false_when_model_files_missing(self, tmp_path):
        """_available is set to False when encoder_model.onnx does not exist."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        # Point _MODEL_DIR at an empty temp directory via module-level patch
        with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)):
            svc._ensure_loaded()

        assert svc._available is False

    def test_is_available_false_when_onnxruntime_not_installed(self, tmp_path):
        """ImportError for onnxruntime sets _available permanently to False."""
        from services.doc2query_service import Doc2QueryService

        # Create stub encoder file so we get past the path check
        (tmp_path / 'encoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_with_past_model.onnx').write_bytes(b'stub')

        svc = Doc2QueryService()
        with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)), \
             patch.dict('sys.modules', {'onnxruntime': None}):
            svc._ensure_loaded()

        assert svc._available is False

    def test_loads_three_sessions_when_files_exist(self, tmp_path):
        """With model files present, exactly 3 InferenceSession calls are made."""
        from services.doc2query_service import Doc2QueryService

        (tmp_path / 'encoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_with_past_model.onnx').write_bytes(b'stub')

        mock_session = MagicMock()
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 3

        svc = Doc2QueryService()
        with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)), \
             patch.dict('sys.modules', {'onnxruntime': mock_ort}), \
             patch('transformers.AutoTokenizer') as mock_tok_cls:
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            svc._ensure_loaded()

        assert mock_ort.InferenceSession.call_count == 3

    def test_loads_tokenizer_from_model_dir(self, tmp_path):
        """AutoTokenizer.from_pretrained is called with the absolute model directory."""
        from services.doc2query_service import Doc2QueryService

        (tmp_path / 'encoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_with_past_model.onnx').write_bytes(b'stub')

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 3

        captured_path = []

        def _from_pretrained(path):
            captured_path.append(path)
            return MagicMock()

        svc = Doc2QueryService()
        with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)), \
             patch.dict('sys.modules', {'onnxruntime': mock_ort}), \
             patch('transformers.AutoTokenizer') as mock_tok_cls:
            mock_tok_cls.from_pretrained.side_effect = _from_pretrained
            svc._ensure_loaded()

        assert len(captured_path) == 1
        assert captured_path[0] == str(tmp_path)


# ---------------------------------------------------------------------------
# 2. Generation
# ---------------------------------------------------------------------------

class TestDoc2QueryGeneration:

    def _make_svc(self, num_queries=2, decode_text="a query"):
        """Build an available Doc2QueryService with all sessions mocked."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = True
        svc._last_used = time.time()
        svc._encoder = _make_encoder_session()
        svc._decoder = _make_decoder_session()
        svc._decoder_past = _make_decoder_past_session()
        svc._tokenizer = _make_tokenizer(decode_text)
        return svc

    def test_generate_queries_returns_list(self):
        """generate_queries() returns a list of strings when sessions are available."""
        svc = self._make_svc(decode_text="example query")
        result = svc.generate_queries("some input text", num_queries=2)
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_generate_queries_returns_empty_when_unavailable(self):
        """generate_queries() returns [] immediately when _available is False."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = False
        result = svc.generate_queries("any text")
        assert result == []
        # is_available must not have tried to reload
        assert svc._encoder is None

    def test_generate_queries_deduplicates_case_insensitive(self):
        """Queries that differ only in case are collapsed to one entry."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = True
        svc._last_used = time.time()
        svc._encoder = _make_encoder_session()
        svc._tokenizer = _make_tokenizer()

        # _generate_one returns a token list; tokenizer.decode converts it
        # Simulate returning same text with different casing across calls
        decode_results = iter([
            "Sister name Elena",
            "sister name elena",
            "sibling of user",
        ])
        svc._tokenizer.decode.side_effect = lambda tokens, **kw: next(decode_results)

        with patch.object(svc, '_generate_one', side_effect=[
            [10, 20, 30],
            [10, 20, 30],
            [40, 50, 60],
        ]):
            result = svc.generate_queries("sister named Elena", num_queries=3)

        # "Sister name Elena" and "sister name elena" are duplicates
        lower_results = [q.lower() for q in result]
        assert lower_results.count("sister name elena") == 1
        assert len(result) == 2

    def test_generate_queries_stops_at_eos(self):
        """When decoder outputs eos_token_id=1, _generate_one returns early."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = True
        svc._last_used = time.time()

        vocab_size = 32128
        num_layers = 2

        # Encoder returns hidden states
        encoder = _make_encoder_session()
        svc._encoder = encoder

        # Decoder step 0 returns token 2
        decoder = _make_decoder_session(vocab_size=vocab_size, num_layers=num_layers)
        svc._decoder = decoder

        # Decoder_past immediately returns EOS (token 1)
        decoder_past = _make_decoder_past_session(vocab_size=vocab_size, num_layers=num_layers)
        svc._decoder_past = decoder_past
        svc._tokenizer = _make_tokenizer("one token query")

        # Force deterministic sampling: token 2 from decoder, then EOS from decoder_past
        with patch.object(
            type(svc), '_sample_top_p',
            staticmethod(lambda logits, **kw: int(np.argmax(logits)))
        ):
            tokens = svc._generate_one(
                np.zeros((1, 5, 512), dtype=np.float32),
                np.ones((1, 5), dtype=np.int64),
                max_length=64,
            )

        # Should have exactly 1 token (token 2 from step 0; step 1 returns EOS which is not appended)
        assert tokens == [2]
        # decoder_past was called exactly once (for step 1)
        assert decoder_past.run.call_count == 1

    def test_generate_queries_respects_max_length(self):
        """Generation stops at max_length tokens even without an EOS token."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = True
        svc._last_used = time.time()

        vocab_size = 100
        num_layers = 1
        max_length = 5

        # Build sessions that never return EOS (token 1)
        encoder = _make_encoder_session(vocab_size=vocab_size)
        svc._encoder = encoder

        # Decoder always returns token 2
        decoder = _make_decoder_session(vocab_size=vocab_size, num_layers=num_layers)
        logits_no_eos = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
        logits_no_eos[0, 0, 2] = 10.0
        kv = [np.zeros((1, 8, 1, 64), dtype=np.float32)]
        decoder.run.return_value = [logits_no_eos] + kv
        svc._decoder = decoder

        # decoder_past also always returns token 2 (no EOS)
        decoder_past = _make_decoder_past_session(vocab_size=vocab_size, num_layers=num_layers)
        decoder_past.run.return_value = [logits_no_eos] + kv
        svc._decoder_past = decoder_past
        svc._tokenizer = _make_tokenizer()

        # Use greedy (argmax) sampling to avoid randomness
        with patch.object(
            type(svc), '_sample_top_p',
            staticmethod(lambda logits, **kw: int(np.argmax(logits)))
        ):
            tokens = svc._generate_one(
                np.zeros((1, 5, 512), dtype=np.float32),
                np.ones((1, 5), dtype=np.int64),
                max_length=max_length,
            )

        assert len(tokens) == max_length


# ---------------------------------------------------------------------------
# 3. KV Cache
# ---------------------------------------------------------------------------

class TestDoc2QueryKVCache:

    def _build_svc(self, num_layers=3):
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = True
        svc._last_used = time.time()
        svc._encoder = _make_encoder_session()
        svc._decoder = _make_decoder_session(num_layers=num_layers)
        svc._decoder_past = _make_decoder_past_session(num_layers=num_layers)
        svc._tokenizer = _make_tokenizer()
        return svc, num_layers

    def test_first_step_uses_decoder_model(self):
        """At step 0, only the decoder session (not decoder_past) is called."""
        svc, _ = self._build_svc()

        with patch.object(
            type(svc), '_sample_top_p',
            staticmethod(lambda logits, **kw: 1)  # always EOS → stops after step 0
        ):
            svc._generate_one(
                np.zeros((1, 5, 512), dtype=np.float32),
                np.ones((1, 5), dtype=np.int64),
                max_length=5,
            )

        svc._decoder.run.assert_called_once()
        svc._decoder_past.run.assert_not_called()

    def test_subsequent_steps_use_decoder_past(self):
        """Steps 1+ always call decoder_past, never decoder again."""
        svc, num_layers = self._build_svc(num_layers=2)

        # Step 0: return token 2; step 1: return token 3; step 2: return EOS
        call_count = [0]
        vocab_size = 32128
        kv = [np.zeros((1, 8, 1, 64), dtype=np.float32) for _ in range(num_layers)]

        def _decoder_past_run(output_names, feeds):
            call_count[0] += 1
            if call_count[0] >= 2:
                # Return EOS on second call
                eos_logits = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
                eos_logits[0, 0, 1] = 10.0
                return [eos_logits] + kv
            # First decoder_past call: return token 3
            tok3_logits = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
            tok3_logits[0, 0, 3] = 10.0
            return [tok3_logits] + kv

        svc._decoder_past.run.side_effect = _decoder_past_run

        with patch.object(
            type(svc), '_sample_top_p',
            staticmethod(lambda logits, **kw: int(np.argmax(logits)))
        ):
            tokens = svc._generate_one(
                np.zeros((1, 5, 512), dtype=np.float32),
                np.ones((1, 5), dtype=np.int64),
                max_length=10,
            )

        # decoder (step-0) called once; decoder_past called for steps 1 and 2
        svc._decoder.run.assert_called_once()
        assert svc._decoder_past.run.call_count == 2
        # Generated tokens: [2, 3] (EOS at step 2 breaks before appending)
        assert tokens == [2, 3]

    def test_encoder_kv_preserved_across_steps(self):
        """KV tensors from step 0 are passed unchanged into decoder_past at step 1."""
        num_layers = 2
        vocab_size = 32128
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._available = True
        svc._last_used = time.time()
        svc._encoder = _make_encoder_session()
        svc._tokenizer = _make_tokenizer()

        # Decoder step 0 returns distinguishable KV sentinel arrays
        sentinel_kv = [
            np.full((1, 8, 1, 64), float(i + 1), dtype=np.float32)
            for i in range(num_layers)
        ]
        logits_tok2 = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
        logits_tok2[0, 0, 2] = 10.0

        decoder = MagicMock()
        decoder.get_outputs.return_value = [_NamedOutput("logits")] + [
            _NamedOutput(f"present.{i}.decoder.key") for i in range(num_layers)
        ]
        decoder.run.return_value = [logits_tok2] + sentinel_kv
        svc._decoder = decoder

        # decoder_past captures its input feeds
        captured_feeds = []
        eos_logits = np.full((1, 1, vocab_size), -100.0, dtype=np.float32)
        eos_logits[0, 0, 1] = 10.0
        fresh_kv = [np.zeros((1, 8, 1, 64), dtype=np.float32) for _ in range(num_layers)]

        decoder_past = MagicMock()
        decoder_past.get_outputs.return_value = [_NamedOutput("logits")] + [
            _NamedOutput(f"present.{i}.decoder.key") for i in range(num_layers)
        ]

        def _past_run(output_names, feeds):
            captured_feeds.append(dict(feeds))
            return [eos_logits] + fresh_kv

        decoder_past.run.side_effect = _past_run
        svc._decoder_past = decoder_past

        with patch.object(
            type(svc), '_sample_top_p',
            staticmethod(lambda logits, **kw: int(np.argmax(logits)))
        ):
            svc._generate_one(
                np.zeros((1, 5, 512), dtype=np.float32),
                np.ones((1, 5), dtype=np.int64),
            )

        assert len(captured_feeds) == 1
        feeds_at_step1 = captured_feeds[0]
        for i in range(num_layers):
            key = f"past_key_values.{i}.decoder.key"
            assert key in feeds_at_step1
            np.testing.assert_array_equal(feeds_at_step1[key], sentinel_kv[i])


# ---------------------------------------------------------------------------
# 4. Top-p Sampling
# ---------------------------------------------------------------------------

class TestDoc2QueryTopPSampling:

    def test_sample_top_p_returns_valid_token_id(self):
        """Output is a non-negative integer within the vocabulary range."""
        from services.doc2query_service import Doc2QueryService

        vocab_size = 1000
        logits = np.random.randn(vocab_size).astype(np.float32)
        token_id = Doc2QueryService._sample_top_p(logits, top_p=0.95)

        assert isinstance(token_id, int)
        assert 0 <= token_id < vocab_size

    def test_sample_top_p_respects_probability_distribution(self):
        """High-probability token is selected significantly more often than others."""
        from services.doc2query_service import Doc2QueryService

        vocab_size = 50
        logits = np.full(vocab_size, -10.0, dtype=np.float32)
        # Token 7 gets overwhelmingly high logit
        logits[7] = 5.0

        np.random.seed(42)
        counts = {}
        for _ in range(200):
            tok = Doc2QueryService._sample_top_p(logits, top_p=0.95)
            counts[tok] = counts.get(tok, 0) + 1

        # Token 7 should win the vast majority of the time
        assert counts.get(7, 0) > 150

    def test_sample_top_p_handles_single_dominant_token(self):
        """When one token has >95% probability, it is always selected."""
        from services.doc2query_service import Doc2QueryService

        vocab_size = 100
        logits = np.full(vocab_size, -50.0, dtype=np.float32)
        # Give token 42 an overwhelming probability mass
        logits[42] = 50.0

        np.random.seed(0)
        results = {Doc2QueryService._sample_top_p(logits, top_p=0.95) for _ in range(50)}
        assert results == {42}


# ---------------------------------------------------------------------------
# 5. Idle Unloading
# ---------------------------------------------------------------------------

class TestDoc2QueryIdleUnloading:

    def test_unload_clears_sessions(self):
        """After _unload(), all sessions are None and _available is None."""
        from services.doc2query_service import Doc2QueryService

        svc = Doc2QueryService()
        svc._encoder = MagicMock()
        svc._decoder = MagicMock()
        svc._decoder_past = MagicMock()
        svc._tokenizer = MagicMock()
        svc._available = True
        svc._last_used = time.time()

        svc._unload()

        assert svc._encoder is None
        assert svc._decoder is None
        assert svc._decoder_past is None
        assert svc._tokenizer is None
        assert svc._available is None
        assert svc._last_used == 0.0

    def test_idle_timeout_triggers_unload(self, tmp_path):
        """_ensure_loaded() unloads sessions when last_used is beyond _IDLE_TIMEOUT."""
        from services.doc2query_service import Doc2QueryService
        import services.doc2query_service as d2q_mod

        svc = Doc2QueryService()
        # Inject a loaded state
        svc._encoder = MagicMock()
        svc._decoder = MagicMock()
        svc._decoder_past = MagicMock()
        svc._tokenizer = MagicMock()
        svc._available = True
        # Set last_used to 11 minutes ago (beyond the 10-minute idle timeout)
        svc._last_used = time.time() - (d2q_mod._IDLE_TIMEOUT + 60)

        # _ensure_loaded with no model files → after unload, available=False
        (tmp_path / 'dummy').write_bytes(b'')
        with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)):
            svc._ensure_loaded()

        # The old sessions should have been cleared via _unload()
        assert svc._encoder is None

    def test_reload_after_unload(self, tmp_path):
        """After _unload(), is_available() re-triggers load attempt."""
        from services.doc2query_service import Doc2QueryService

        (tmp_path / 'encoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_with_past_model.onnx').write_bytes(b'stub')

        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 3

        svc = Doc2QueryService()

        with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)), \
             patch.dict('sys.modules', {'onnxruntime': mock_ort}), \
             patch('transformers.AutoTokenizer') as mock_tok_cls:
            mock_tok_cls.from_pretrained.return_value = MagicMock()

            # First load
            svc._ensure_loaded()
            assert svc._available is True

            # Unload
            svc._unload()
            assert svc._available is None

            # Re-check: is_available() should re-trigger _ensure_loaded
            result = svc.is_available()

        assert result is True
        # Three sessions created twice (load + reload)
        assert mock_ort.InferenceSession.call_count == 6


# ---------------------------------------------------------------------------
# 6. Thread Safety
# ---------------------------------------------------------------------------

class TestDoc2QueryThreadSafety:

    def test_concurrent_loads_create_single_session(self, tmp_path):
        """Multiple threads racing into _ensure_loaded() produce exactly one set of sessions."""
        from services.doc2query_service import Doc2QueryService

        (tmp_path / 'encoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_model.onnx').write_bytes(b'stub')
        (tmp_path / 'decoder_with_past_model.onnx').write_bytes(b'stub')

        created_sessions = []

        def _make_session(*args, **kwargs):
            session = MagicMock()
            created_sessions.append(session)
            return session

        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = _make_session
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 3

        svc = Doc2QueryService()

        barrier = threading.Barrier(8)
        errors = []

        def _load():
            try:
                barrier.wait(timeout=5)
                with patch('services.doc2query_service._MODEL_DIR', str(tmp_path)), \
                     patch.dict('sys.modules', {'onnxruntime': mock_ort}), \
                     patch('transformers.AutoTokenizer') as mock_tok_cls:
                    mock_tok_cls.from_pretrained.return_value = MagicMock()
                    svc._ensure_loaded()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_load) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        # Exactly 3 sessions (encoder + decoder + decoder_past) regardless of thread count
        assert len(created_sessions) == 3
