"""Probability helpers for at-bat outcome simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from prediction_models.prediction import predict_auto

_DEFAULT_MODEL_OUTPUT = [0.15, 0.05, 0.01, 0.03, 0.76]
_INFIELD_FLY_RATIO = 0.13


@dataclass
class OutcomeProbabilityCalculator:
    """Builds probability distributions for at-bat outcomes.

    responsibility: Weight of batter vs pitcher contribution in component
    probabilities. In [0,1]. 0.5 means equal (current behavior), values >0.5
    weight batter more, values <0.5 weight pitcher more.
    """

    model: Optional[object]
    model_type: str
    responsibility: float = 0.5

    def calculate(self, batter, pitcher) -> Dict[str, float]:
        """Return a probability distribution over possible at-bat results."""
        k_prob = self._calculate_component(
            batter.k_pct, pitcher.k_pct, league_average=22.8, responsibility=0.65
        )
        bb_prob = self._calculate_component(
            batter.bb_pct, pitcher.bb_pct, league_average=8.5, responsibility=0.65
        )
        hard_prob = self._calculate_component(
            batter.hard_pct, pitcher.hard_pct, league_average=38.6, responsibility=0.8
        )
        gb_prob = self._calculate_component(
            batter.gb_pct, pitcher.gb_pct, league_average=44.6, responsibility=0.5
        )

        other_prob = max(0.0, 1 - k_prob - bb_prob)
        model_prediction = self._get_model_prediction({
            "K%": k_prob,
            "BB%": bb_prob,
            "Hard%": hard_prob,
            "GB%": gb_prob,
        })

        # NNの出力が4要素（1B,2B,3B,HR）または5要素（+OTH）どちらにも対応。
        # single_prob などは最小値を0にし、total_hits が other_prob を超える場合のみスケーリング。
        single_prob = double_prob = triple_prob = hr_prob = out_woSO_prob = 0.0
        cleaned = list(model_prediction) if isinstance(model_prediction, (list, tuple)) else []
        if len(cleaned) >= 4:
            single_prob = max(0.0, float(cleaned[0]))
            double_prob = max(0.0, float(cleaned[1]))
            triple_prob = max(0.0, float(cleaned[2]))
            hr_prob = max(0.0, float(cleaned[3]))

            total_hits = single_prob + double_prob + triple_prob + hr_prob
            if total_hits > other_prob and total_hits > 0:
                scale = other_prob / total_hits
                single_prob = max(0.0, single_prob * scale)
                double_prob = max(0.0, double_prob * scale)
                triple_prob = max(0.0, triple_prob * scale)
                hr_prob = max(0.0, hr_prob * scale)

            out_woSO_prob = max(0.0, other_prob - (single_prob + double_prob + triple_prob + hr_prob))
        else:
            # 予期しない形状はデフォルトにフォールバック
            single_prob, double_prob, triple_prob, hr_prob, out_woSO_prob = _DEFAULT_MODEL_OUTPUT

        groundout_prob = out_woSO_prob * gb_prob
        flyout_prob = max(0.0, out_woSO_prob - groundout_prob)
        infield_flyout_prob = flyout_prob * _INFIELD_FLY_RATIO
        outfield_flyout_prob = max(0.0, flyout_prob - infield_flyout_prob)

        return {
            "strikeout": k_prob,
            "walk": bb_prob,
            "single": single_prob,
            "double": double_prob,
            "triple": triple_prob,
            "home_run": hr_prob,
            "groundout": groundout_prob,
            "infield_flyout": infield_flyout_prob,
            "outfield_flyout": outfield_flyout_prob,
        }

    def _calculate_component(
        self,
        batter_value: float,
        pitcher_value: float,
        league_average: float,
        responsibility: float = 0.5,
    ) -> float:
        """Combine batter/pitcher component into a single probability.

        This uses a weighted odds-combination. Let prior odds be
        (1-L)/L where L is league rate. Let LR_b = b/(1-b), LR_p = p/(1-p).
        Then posterior odds = prior_odds * LR_b^(2r) * LR_p^(2(1-r)), where
        r in [0,1] is the responsibility weight (r=0.5 -> original behavior).
        """
        # Clamp responsibility to [0,1] for safety
        r = max(0.0, min(1.0, responsibility))
        # Convert percentages to rates
        league_rate = league_average / 100.0
        batter_rate = batter_value / 100.0
        pitcher_rate = pitcher_value / 100.0

        # Guard against degenerate league rates
        eps = 1e-12
        league_rate = min(max(league_rate, eps), 1.0 - eps)
        batter_rate = min(max(batter_rate, 0.0), 1.0)
        pitcher_rate = min(max(pitcher_rate, 0.0), 1.0)

        # If any side is exactly 0 and the other weighting would nullify it at r=0.5,
        # return 0 to mirror the original behavior.
        if r == 0.5 and (batter_rate == 0.0 or pitcher_rate == 0.0):
            return 0.0

        # Avoid division by zero in likelihood ratios by clamping away from 0/1
        b = min(max(batter_rate, eps), 1.0 - eps)
        p = min(max(pitcher_rate, eps), 1.0 - eps)

        # Likelihood ratios
        lr_b = b / (1.0 - b)
        lr_p = p / (1.0 - p)
        prior_odds = (1.0 - league_rate) / league_rate

        # Weighted combination; exponents sum to 2 so r=0.5 recovers original (power 1 each)
        eb = 2.0 * r
        ep = 2.0 * (1.0 - r)
        posterior_odds = prior_odds * (lr_b ** eb) * (lr_p ** ep)

        # Convert odds back to probability
        prob = posterior_odds / (1.0 + posterior_odds)
        # Numerical safety
        if prob < 0.0:
            return 0.0
        if prob > 1.0:
            return 1.0
        return prob

    def _get_model_prediction(self, data) -> list:
        if self.model is None:
            return _DEFAULT_MODEL_OUTPUT.copy()

        prediction = predict_auto(self.model, data, self.model_type)
        cleaned = []
        for value in prediction:
            cleaned.append(max(0.0, value))
        return cleaned
