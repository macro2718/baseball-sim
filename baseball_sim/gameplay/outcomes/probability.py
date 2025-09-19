"""Probability helpers for at-bat outcome simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from prediction_models.prediction import predict_auto

_DEFAULT_MODEL_OUTPUT = [0.15, 0.05, 0.01, 0.03, 0.76]


@dataclass
class OutcomeProbabilityCalculator:
    """Builds probability distributions for at-bat outcomes."""

    model: Optional[object]
    model_type: str

    def calculate(self, batter, pitcher) -> Dict[str, float]:
        """Return a probability distribution over possible at-bat results."""
        k_prob = self._calculate_component(batter.k_pct, pitcher.k_pct, league_average=22.8)
        bb_prob = self._calculate_component(batter.bb_pct, pitcher.bb_pct, league_average=8.5)
        hard_prob = self._calculate_component(batter.hard_pct, pitcher.hard_pct, league_average=38.6)
        gb_prob = self._calculate_component(batter.gb_pct, pitcher.gb_pct, league_average=44.6)

        other_prob = max(0.0, 1 - k_prob - bb_prob)
        model_prediction = self._get_model_prediction({
            "K%": k_prob,
            "BB%": bb_prob,
            "Hard%": hard_prob,
            "GB%": gb_prob,
        })

        single_prob, double_prob, triple_prob, hr_prob, out_woSO_prob = model_prediction
        total = single_prob + double_prob + triple_prob + hr_prob + out_woSO_prob
        if total > 0:
            scale = other_prob / total
            single_prob *= scale
            double_prob *= scale
            triple_prob *= scale
            hr_prob *= scale
            out_woSO_prob *= scale
        else:
            single_prob = double_prob = triple_prob = hr_prob = 0.0
            out_woSO_prob = other_prob

        groundout_prob = out_woSO_prob * gb_prob
        flyout_prob = max(0.0, out_woSO_prob - groundout_prob)

        return {
            "strikeout": k_prob,
            "walk": bb_prob,
            "single": single_prob,
            "double": double_prob,
            "triple": triple_prob,
            "home_run": hr_prob,
            "groundout": groundout_prob,
            "flyout": flyout_prob,
        }

    def _calculate_component(self, batter_value: float, pitcher_value: float, league_average: float) -> float:
        league_rate = league_average / 100
        batter_rate = batter_value / 100
        pitcher_rate = pitcher_value / 100
        if batter_rate * pitcher_rate == 0:
            return 0.0
        numerator = batter_rate * pitcher_rate * (1 - league_rate)
        denominator = numerator + (1 - batter_rate) * (1 - pitcher_rate) * league_rate
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _get_model_prediction(self, data) -> list:
        if self.model is None:
            return _DEFAULT_MODEL_OUTPUT.copy()

        prediction = predict_auto(self.model, data, self.model_type)
        cleaned = []
        for value in prediction:
            cleaned.append(max(0.0, value))
        if len(cleaned) < 5:
            cleaned.extend([0.0] * (5 - len(cleaned)))
        return cleaned[:5]
