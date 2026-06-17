
import logging
import math
import os
import threading
import time
from typing import List

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[DOC2QUERY]"
_ONNX_PRESENT_PREFIX = "present."
_MODEL_DIR = str(FileMapperService.get_models_path("doc2query-small"))
_IDLE_TIMEOUT = 600  # 10 minutes


class Doc2QueryService:

    def __init__(self):
        self._encoder = None
        self._decoder = None
        self._decoder_past = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._available = None
        self._last_used = 0.0

    def is_available(self) -> bool:
        if self._available is None:
            self._ensure_loaded()
        return self._available or False

    def _ensure_loaded(self):
        with self._lock:
            # Check idle timeout under lock to prevent race with generate_queries
            if self._encoder is not None and self._last_used > 0:
                if time.time() - self._last_used > _IDLE_TIMEOUT:
                    self._unload_unlocked()

            if self._encoder is not None:
                return
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                from services.onnx_session import build_session

                model_dir = os.path.abspath(_MODEL_DIR)
                encoder_path = os.path.join(model_dir, 'encoder_model.onnx')
                decoder_path = os.path.join(model_dir, 'decoder_model.onnx')
                decoder_past_path = os.path.join(model_dir, 'decoder_with_past_model.onnx')

                missing = [p for p in (encoder_path, decoder_path, decoder_past_path)
                           if not os.path.exists(p)]
                if missing:
                    names = [os.path.basename(p) for p in missing]
                    logger.info(f"{LOG_PREFIX} Missing model files at %s: %s", model_dir, ', '.join(names))
                    self._available = False
                    return

                def _opts():
                    o = ort.SessionOptions()
                    o.intra_op_num_threads = 1
                    o.inter_op_num_threads = 1
                    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    return o

                self._encoder = build_session(encoder_path, sess_options=_opts(), log_prefix=LOG_PREFIX)
                self._decoder = build_session(decoder_path, sess_options=_opts(), log_prefix=LOG_PREFIX)
                self._decoder_past = build_session(decoder_past_path, sess_options=_opts(), log_prefix=LOG_PREFIX)
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

                self._available = True
                self._last_used = time.time()
                logger.info(f"{LOG_PREFIX} Model loaded (3 ONNX sessions)")
            except ImportError as e:
                logger.warning(f"{LOG_PREFIX} Permanently unavailable (missing dependency): %s", e)
                self._available = False
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Load failed (will retry): %s", e)
                # Reset partial state so retry can re-attempt from scratch
                self._encoder = None
                self._decoder = None
                self._decoder_past = None
                self._tokenizer = None
                self._available = None

    def _unload_unlocked(self):
        self._encoder = None
        self._decoder = None
        self._decoder_past = None
        self._tokenizer = None
        self._available = None
        self._last_used = 0.0
        logger.info(f"{LOG_PREFIX} Unloaded (idle timeout)")

    def _unload(self):
        with self._lock:
            self._unload_unlocked()

    def generate_queries(self, text: str, num_queries: int = 5) -> List[str]:
        if not self.is_available():
            return []
        self._last_used = time.time()

        try:
            encoded = self._tokenizer(text, max_length=384, truncation=True, return_tensors="np")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            encoder_out = self._encoder.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            encoder_hidden = encoder_out[0]

            queries = []
            for _ in range(num_queries):
                tokens = self._generate_one(encoder_hidden, attention_mask)
                text_out = self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if text_out:
                    queries.append(text_out)

            seen = set()
            unique = []
            for q in queries:
                low = q.lower()
                if low not in seen:
                    seen.add(low)
                    unique.append(q)
            return unique
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Generation failed: %s", e)
            return []

    def _generate_one(self, encoder_hidden, attention_mask, max_length=64):
        import numpy as np

        decoder_ids = np.array([[0]], dtype=np.int64)
        past_kv = None
        generated = []

        for step in range(max_length):
            if step == 0:
                feeds = {
                    "input_ids": decoder_ids,
                    "encoder_attention_mask": attention_mask,
                    "encoder_hidden_states": encoder_hidden,
                }
                outputs = self._decoder.run(None, feeds)
                logits = outputs[0]
                output_names = [o.name for o in self._decoder.get_outputs()]
                past_kv = {}
                for i, name in enumerate(output_names):
                    if name.startswith(_ONNX_PRESENT_PREFIX):
                        past_name = name.replace(_ONNX_PRESENT_PREFIX, "past_key_values.", 1)
                        past_kv[past_name] = outputs[i]
            else:
                feeds = {
                    "input_ids": decoder_ids,
                    "encoder_attention_mask": attention_mask,
                }
                feeds.update(past_kv)
                outputs = self._decoder_past.run(None, feeds)
                logits = outputs[0]
                output_names = [o.name for o in self._decoder_past.get_outputs()]
                for i, name in enumerate(output_names):
                    if name.startswith(_ONNX_PRESENT_PREFIX):
                        past_name = name.replace(_ONNX_PRESENT_PREFIX, "past_key_values.", 1)
                        past_kv[past_name] = outputs[i]

            next_token = self._sample_top_p(logits[0, -1, :], top_p=0.95)

            if next_token == 1:
                break

            generated.append(next_token)
            decoder_ids = np.array([[next_token]], dtype=np.int64)

        return generated

    @staticmethod
    def _sample_top_p(logits, top_p=0.95, temperature=1.0):
        import numpy as np

        if not math.isclose(temperature, 1.0):
            logits = logits / temperature

        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        cumsum = np.cumsum(sorted_probs)
        mask = cumsum - sorted_probs <= top_p

        filtered_probs = sorted_probs * mask
        filtered_probs /= filtered_probs.sum()

        chosen_idx = np.random.default_rng().choice(len(filtered_probs), p=filtered_probs)  # NOSONAR
        return int(sorted_indices[chosen_idx])


_instance = None
_instance_lock = threading.Lock()


def get_doc2query_service() -> Doc2QueryService:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Doc2QueryService()
    return _instance
