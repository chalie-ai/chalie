"""Generate search queries for knowledge entries using doc2query T5 model."""

import logging
import os
import threading
from typing import List

logger = logging.getLogger(__name__)

_MODEL_NAME = "doc2query/msmarco-t5-small-v1"
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'doc2query-small')


class Doc2QueryService:
    """Generates potential search queries for knowledge entries using a dedicated T5 model."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._available = None  # None = not checked yet

    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        if self._available is None:
            self._ensure_loaded()
        return self._available or False

    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from optimum.onnxruntime import ORTModelForSeq2SeqLM
                from transformers import AutoTokenizer

                model_dir = os.path.abspath(_MODEL_DIR)

                # Check if ONNX model exists; if not, export from HuggingFace
                if not os.path.exists(os.path.join(model_dir, 'encoder_model.onnx')):
                    logger.info(f"[DOC2QUERY] Exporting {_MODEL_NAME} to ONNX at {model_dir}...")
                    os.makedirs(model_dir, exist_ok=True)
                    model = ORTModelForSeq2SeqLM.from_pretrained(
                        _MODEL_NAME, export=True
                    )
                    model.save_pretrained(model_dir)
                    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
                    tokenizer.save_pretrained(model_dir)
                    self._model = model
                    self._tokenizer = tokenizer
                else:
                    self._model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
                    self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

                self._available = True
                logger.info("[DOC2QUERY] Model loaded successfully")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[DOC2QUERY] Model permanently unavailable (missing dependency): {e}")
                self._available = False
            except Exception as e:
                logger.warning(f"[DOC2QUERY] Model load failed (will retry): {e}")
                self._available = None  # transient — allow retry on next call

    def generate_queries(self, text: str, num_queries: int = 5) -> List[str]:
        """Generate potential search queries for a knowledge entry text.

        Args:
            text: The knowledge entry text (e.g., "sister: Elena")
            num_queries: Number of queries to generate

        Returns:
            List of generated query strings. Empty list if model unavailable.
        """
        if not self.is_available():
            return []

        try:
            inputs = self._tokenizer(
                text, max_length=384, truncation=True, return_tensors="pt"
            )

            outputs = self._model.generate(
                **inputs,
                max_length=64,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries,
            )

            queries = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for q in queries:
                q = q.strip()
                if q and q.lower() not in seen:
                    seen.add(q.lower())
                    unique.append(q)
            return unique
        except Exception as e:
            logger.warning(f"[DOC2QUERY] Generation failed: {e}")
            return []


# Singleton
_instance = None
_instance_lock = threading.Lock()


def get_doc2query_service() -> Doc2QueryService:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Doc2QueryService()
    return _instance
