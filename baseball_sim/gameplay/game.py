"""
野球ゲームの試合進行を管理するメインモジュール
バッティング予測、守備処理、ゲーム状態管理を担当
"""
from __future__ import annotations

import random

from baseball_sim.config import (
    INNINGS_PER_GAME,
    MAX_EXTRA_INNINGS,
    OUTS_PER_INNING,
    config,
    path_manager,
)
from baseball_sim.gameplay.outcomes.handler import AtBatOutcomeHandler
from baseball_sim.gameplay.outcomes.probability import OutcomeProbabilityCalculator
from baseball_sim.gameplay.state import (
    BaseRunners,
    DefensiveStatus,
    HalfInning,
    HalfInningState,
    Scoreboard,
)
from baseball_sim.prediction.batting import BattingModelLoader

try:
    from baseball_sim.infrastructure.logging_utils import logger
except ImportError:  # pragma: no cover - fallback for environments without infra module
    class logger:  # type: ignore
        @staticmethod
        def info(msg):
            print(f"INFO: {msg}")

        @staticmethod
        def warning(msg):
            print(f"WARNING: {msg}")

        @staticmethod
        def error(msg):
            print(f"ERROR: {msg}")


class GameState:
    """Central game controller tracking score, bases, innings, and outcomes."""

    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team

        self._half_state = HalfInningState(outs_per_inning=OUTS_PER_INNING)
        self._bases = BaseRunners()
        self._scoreboard = Scoreboard()
        self._defense_status = DefensiveStatus()

        self._scoreboard.open_new_inning()

        self.batting_team = away_team  # アウェイチームが先攻
        self.fielding_team = home_team
        self.game_ended = False

        model_info = BattingModelLoader(config, path_manager, logger).load()
        self.model = model_info.estimator
        self.model_type = model_info.model_type
        self._probability_calculator = OutcomeProbabilityCalculator(self.model, self.model_type)
        self._outcome_handler = AtBatOutcomeHandler(self)

    # ------------------------------------------------------------------
    # Properties exposing internal state in a backward compatible manner
    # ------------------------------------------------------------------
    @property
    def inning(self) -> int:
        return self._half_state.inning

    @property
    def is_top_inning(self) -> bool:
        return self._half_state.is_top

    @property
    def outs(self) -> int:
        return self._half_state.outs

    @outs.setter
    def outs(self, value: int) -> None:
        self._half_state.outs = value

    @property
    def bases(self) -> BaseRunners:
        return self._bases

    @bases.setter
    def bases(self, value) -> None:
        self._bases = BaseRunners.from_value(value)

    @property
    def home_score(self) -> int:
        return self._scoreboard.home_total

    @home_score.setter
    def home_score(self, value: int) -> None:
        self._scoreboard.home_total = value

    @property
    def away_score(self) -> int:
        return self._scoreboard.away_total

    @away_score.setter
    def away_score(self, value: int) -> None:
        self._scoreboard.away_total = value

    @property
    def inning_scores(self):
        return self._scoreboard.inning_scores

    @property
    def defensive_position_error(self) -> bool:
        return self._defense_status.frozen

    @property
    def defensive_error_messages(self):
        return list(self._defense_status.messages)

    # ------------------------------------------------------------------
    # Core game flow
    # ------------------------------------------------------------------
    def switch_sides(self):
        """攻守交代の処理"""
        self.batting_team.next_batter()

        if self.is_top_inning:
            if self._complete_top_half():
                return
        else:
            if self._complete_bottom_half():
                return

        self._bases.clear()
        self._half_state.reset_outs()
        self._evaluate_defensive_alignment()

    def _complete_top_half(self) -> bool:
        """Handle transition from top to bottom of an inning."""
        if self.inning >= INNINGS_PER_GAME and self.home_score > self.away_score:
            self.game_ended = True
            return True

        self._half_state.move_to_bottom()
        self.batting_team = self.home_team
        self.fielding_team = self.away_team
        return False

    def _complete_bottom_half(self) -> bool:
        """Handle transition from bottom to the next inning."""
        if self.inning >= INNINGS_PER_GAME and self.home_score != self.away_score:
            self.game_ended = True
            return True

        if self.inning >= MAX_EXTRA_INNINGS:
            self.game_ended = True
            logger.info(f"Game ended due to extra innings limit (inning {self.inning})")
            return True

        self._half_state.move_to_top_of_next_inning()
        self._scoreboard.open_new_inning()
        self.batting_team = self.away_team
        self.fielding_team = self.home_team
        return False

    def _evaluate_defensive_alignment(self) -> None:
        if self.fielding_team is None:
            self._defense_status.clear()
            return

        is_ready, messages = self.fielding_team.check_defensive_readiness()
        if not is_ready:
            self._defense_status.freeze(messages)
            print(f"\n⚠️ DEFENSIVE POSITION ERRORS for {self.fielding_team.name}:")
            for error in messages:
                print(f"   ❌ {error}")
            print("   🚫 GAME ACTIONS FROZEN - Please fix defensive positions before continuing play.")
        elif messages:
            self._defense_status.warn(messages)
            print(f"\n⚠️ Defensive Position Warnings for {self.fielding_team.name}:")
            for warning in messages:
                print(f"   ⚠️ {warning}")
        else:
            self._defense_status.clear()

    def is_game_action_allowed(self):
        """ゲームアクション（バッティングなど）が実行可能かどうかを判定"""
        if self.game_ended:
            return False, "Game has ended"

        if self._defense_status.frozen:
            error_msg = "Game actions are frozen due to defensive position errors: " \
                + ", ".join(self._defense_status.messages)
            return False, error_msg

        return True, ""

    def add_out(self):
        """アウトカウントを増やし、必要に応じて攻守交代"""
        inning_complete = self._half_state.register_out()
        if inning_complete:
            self.switch_sides()

    def can_bunt(self):
        """バントが可能かどうかを判定"""
        if self._bases.is_empty():
            return False
        if self.outs >= 2:
            return False
        return True

    def execute_bunt(self, batter, pitcher):
        """バントを実行し、結果を返す"""
        from baseball_sim.gameplay.utils import BuntProcessor

        bunt_processor = BuntProcessor(self)
        return bunt_processor.execute(batter, pitcher)

    def _add_runs(self, runs, batter):
        """得点の加算処理"""
        if runs <= 0:
            return

        is_home = self.batting_team == self.home_team
        inning_index = self.inning - 1
        self._scoreboard.add_runs(is_home=is_home, inning_index=inning_index, runs=runs)

        # サヨナラ勝ち判定
        if (
            is_home
            and self.inning >= INNINGS_PER_GAME
            and not self.is_top_inning
            and self.home_score > self.away_score
        ):
            self.game_ended = True

        self.fielding_team.current_pitcher.pitching_stats["ER"] += runs
        batter.stats["RBI"] += runs

    def calculate_result(self, batter, pitcher):
        """セイバーメトリクスに基づいた打席結果計算"""
        probabilities = self._probability_calculator.calculate(batter, pitcher)
        outcomes = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(outcomes, weights=weights, k=1)[0]

    def apply_result(self, result, batter):
        """結果をゲーム状態に適用し、メッセージを返す"""
        batter.stats["PA"] += 1
        return self._outcome_handler.apply(result, batter)
