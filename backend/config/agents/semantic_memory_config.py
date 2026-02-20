"""
Semantic Memory Configuration - Type-safe config access for semantic memory.

Bridges naming inconsistencies and provides typed access to config values.
"""

from typing import Dict, Any
from services.config_service import ConfigService


class SemanticMemoryConfig:
    """
    Type-safe configuration wrapper for semantic-memory.json.

    Bridges naming inconsistencies:
    - inference_weights → retrieval_weights (unified naming)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize config wrapper.

        Args:
            config: Optional config dict. If not provided, loads semantic-memory.json
        """
        self._config = config or ConfigService.get_agent_config("semantic-memory")

    @property
    def embedding_model(self) -> str:
        """Get embedding model name (sentence-transformers)."""
        return self._config.get('embedding_model', 'all-mpnet-base-v2')

    @property
    def embedding_dimensions(self) -> int:
        """Get embedding vector dimensions."""
        return self._config.get('embedding_dimensions', 256)

    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold for fuzzy matching."""
        return self._config.get('similarity_threshold', 0.85)

    @property
    def min_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for retrieval."""
        return self._config.get('min_confidence_threshold', 0.4)

    @property
    def retrieval_weights(self) -> Dict[str, float]:
        """
        Get retrieval weights (unified naming).

        Bridges 'inference_weights' in config → 'retrieval_weights' in code.
        """
        return self._config.get('inference_weights', {
            'vector_similarity': 5,
            'strength': 3,
            'activation_score': 2,
            'utility_score': 2,
            'confidence': 1
        })

    @property
    def inference_weights(self) -> Dict[str, float]:
        """Get inference weights (alias for retrieval_weights)."""
        return self.retrieval_weights

    @property
    def reinforcement_factor(self) -> float:
        """Get reinforcement factor for concept strengthening."""
        return self._config.get('reinforcement_factor', 0.1)

    @property
    def max_strength(self) -> float:
        """Get maximum strength value."""
        return self._config.get('max_strength', 10.0)

    @property
    def min_strength_floor(self) -> float:
        """Get minimum strength floor (decay limit)."""
        return self._config.get('min_strength_floor', 0.2)

    @property
    def base_decay_rate(self) -> float:
        """Get base decay rate per day."""
        return self._config.get('base_decay_rate', 0.05)

    @property
    def activation_threshold(self) -> float:
        """Get activation threshold for spreading activation."""
        return self._config.get('activation_threshold', 0.3)

    @property
    def max_spreading_depth(self) -> int:
        """Get maximum depth for spreading activation."""
        return self._config.get('max_spreading_depth', 2)

    @property
    def decay_factor(self) -> float:
        """Get decay factor for spreading activation."""
        return self._config.get('decay_factor', 0.7)

    @property
    def weak_relationship_random_activation(self) -> float:
        """Get probability for weak relationships to activate (creative leaps)."""
        return self._config.get('weak_relationship_random_activation', 0.15)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get raw config value.

        Args:
            key: Config key
            default: Default value if key not found

        Returns:
            Config value or default
        """
        return self._config.get(key, default)

    def get_database_config(self) -> Dict[str, Any]:
        """Get database pool configuration."""
        return self._config.get('database', {
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30
        })
