import logging
import time

import requests
import json
import ollama
from services.llm_service import LLMResponse, RateLimitError

class OllamaService:

    def __init__(self, config: dict):
        platform = config.get('platform', 'ollama')
        if platform != 'ollama':
            raise ValueError(f"OllamaService does not support platform '{platform}'")

        self._config = config
        self.host = config.get('host')
        self.model = config.get('model')
        self.keep_alive = config.get('keep_alive', '0')
        self.temperature = config.get('temperature', 0.5)
        self.timeout = config.get('timeout', 60)
        self.format = config.get('format', 'json')
        self.max_retries = config.get('max_retries', 2)

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message to Ollama and return the response."""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": user_message,
            "system": system_prompt,
            "stream": False,
            "think": False,
            "raw": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            }
        }

        # Only add format if not "text" (Ollama treats omission as natural language)
        if self.format != "text":
            payload["format"] = self.format

        last_exception = None
        for attempt in range(1 + self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    text=data['response'],
                    model=data.get('model', self.model),
                    provider='ollama',
                    tokens_input=data.get('prompt_eval_count'),
                    tokens_output=data.get('eval_count'),
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    backoff = 2 * (2 ** attempt)
                    logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after {type(e).__name__}: {e} — backoff {backoff}s")
                    time.sleep(backoff)
                else:
                    logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed: {e}")
                    raise
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    retry_after = None
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
                    raise RateLimitError(str(e), retry_after=retry_after, provider='ollama') from e
                elif e.response is not None and e.response.status_code >= 500:
                    last_exception = e
                    if attempt < self.max_retries:
                        backoff = 1.5 * (2 ** attempt)
                        logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after HTTP {e.response.status_code} — backoff {backoff}s")
                        time.sleep(backoff)
                    else:
                        logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed: {e}")
                        raise
                else:
                    raise

    def generate_embedding(self, text: str, embedding_model: str = None, target_dimensions: int = None) -> list:
        """
        Generate embedding vector using sentence-transformers (no Ollama required).

        Note: embedding_model and target_dimensions parameters are deprecated and ignored.
        All embeddings now use the unified EmbeddingService.

        Args:
            text: Text to embed
            embedding_model: (deprecated, ignored)
            target_dimensions: (deprecated, ignored)

        Returns:
            Embedding vector (768-dim, L2-normalized)

        Raises:
            Exception if embedding generation fails
        """
        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            return emb_service.generate_embedding(text)

        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            raise
