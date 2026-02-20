"""
Salience Service - Computational salience scoring from LLM-provided factors.
Responsibility: Apply weighted formula to salience factors from LLM.
"""

import logging


class SalienceService:
    """Computes salience scores from LLM-provided factors."""

    def __init__(self, config: dict = None):
        """
        Initialize salience service.

        Args:
            config: Optional config dict (currently unused, weights are fixed)
        """
        self.config = config or {}
        logging.info("SalienceService initialized with fixed weights: novelty=0.4, emotional=0.4, commitment=0.2")

    def calculate_salience(self, salience_factors: dict) -> float:
        """
        Calculate salience score from LLM-provided factors.

        Formula:
        - base = 0.4·novelty + 0.4·emotional + 0.2·commitment
        - if unresolved: base *= 1.25
        - salience = clamp(base, 0.1, 1.0)

        Args:
            salience_factors: Dict with keys: novelty (0-3), emotional (0-3), commitment (0-3), unresolved (bool)

        Returns:
            Salience score in range [0.1, 1.0], rounded to 3 decimals
        """
        try:
            # Extract factors and normalize from 0-3 scale to 0-1 scale
            novelty_n = float(salience_factors.get('novelty', 0)) / 3.0
            emotional_n = float(salience_factors.get('emotional', 0)) / 3.0
            commitment_n = float(salience_factors.get('commitment', 0)) / 3.0
            unresolved = salience_factors.get('unresolved', False)

            # Weighted base salience
            base = (
                0.4 * novelty_n +
                0.4 * emotional_n +
                0.2 * commitment_n
            )

            # Unresolved episodes resist decay more strongly
            if unresolved:
                base *= 1.25

            # Clamp and floor
            salience = max(0.1, min(base, 1.0))

            logging.debug(
                f"Calculated salience: {salience:.3f} "
                f"(N={novelty_n:.3f}, E={emotional_n:.3f}, C={commitment_n:.3f}, U={unresolved})"
            )

            return round(salience, 3)

        except Exception as e:
            logging.error(f"Failed to calculate salience: {e}")
            # Return neutral salience on error
            return 0.5
