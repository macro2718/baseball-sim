"""
野球ゲームの試合進行を管理するメインモジュール
バッティング予測、守備処理、ゲーム状態管理を担当
"""
from __future__ import annotations

import random

from baseball_sim.config import (
    BuntConstants,
    GameResults,
    INNINGS_PER_GAME,
    MAX_EXTRA_INNINGS,
    OUTS_PER_INNING,
    StatColumns,
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
        self._play_sequence = 0
        self.last_play = {"result": None, "message": "", "sequence": 0}

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
    def defensive_error_messages(self):
        return list(self._defense_status.messages)

    # ------------------------------------------------------------------
    # Core game flow
    # ------------------------------------------------------------------
    def switch_sides(self, advance_lineup: bool = True):
        """攻守交代の処理"""
        if advance_lineup:
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

    def add_out(self, credit_pitcher: bool = True, advance_lineup: bool = True):
        """アウトカウントを増やし、必要なら投手のIPを更新する"""

        if credit_pitcher:
            pitching_stats = getattr(
                getattr(self.fielding_team, "current_pitcher", None), "pitching_stats", None
            )
            if isinstance(pitching_stats, dict):
                pitching_stats["IP"] = pitching_stats.get("IP", 0) + (1 / 3)

        inning_complete = self._half_state.register_out()
        if inning_complete:
            self.switch_sides(advance_lineup=advance_lineup)

    def can_bunt(self):
        """バントが可能かどうかを判定（走者あり かつ 2アウト未満）"""
        if self._bases.is_empty():
            return False
        if self.outs >= 2:
            return False
        return True

    def can_squeeze(self):
        """スクイズが可能かどうかを判定（三塁走者あり かつ 2アウト未満）"""
        if self.outs >= 2:
            return False
        return bool(self._bases[2])

    def can_steal(self) -> bool:
        """盗塁（またはダブルスチール）の可能性があるか判定"""
        first_runner = self._bases[0]
        second_runner = self._bases[1]
        third_runner = self._bases[2]

        if first_runner and not second_runner:
            return True
        if second_runner and not third_runner:
            return True
        if first_runner and second_runner and not third_runner:
            return True
        return False

    def record_play(self, result, message):
        """プレー結果を記録し連番カウンタを進める"""

        result_key = str(result) if result is not None else None
        message_text = str(message) if message is not None else ""
        self._play_sequence += 1
        self.last_play = {
            "result": result_key,
            "message": message_text,
            "sequence": self._play_sequence,
        }

    def execute_bunt(self, batter, pitcher):
        """バント処理を実行し結果メッセージを返す"""
        from baseball_sim.gameplay.utils import BuntProcessor

        bunt_processor = BuntProcessor(self)
        return bunt_processor.execute(batter, pitcher)

    def execute_squeeze(self, batter, pitcher):
        """スクイズ処理を実行し結果メッセージを返す"""
        from baseball_sim.gameplay.utils import SqueezeProcessor

        squeeze_processor = SqueezeProcessor(self)
        return squeeze_processor.execute(batter, pitcher)

    def _steal_success_probability(self, runner) -> float:
        base_rate = 0.7
        average_speed = getattr(BuntConstants, "STANDARD_RUNNER_SPEED", 100.0) or 100.0
        runner_speed = getattr(runner, "speed", average_speed) or average_speed
        speed_factor = runner_speed / average_speed if average_speed else 1.0
        probability = base_rate * speed_factor
        return max(0.3, min(0.95, probability))

    def _register_steal_attempt(self, runner, success: bool) -> None:
        if runner is None:
            return
        stats = getattr(runner, "stats", None)
        if not isinstance(stats, dict):
            return
        attempts_key = StatColumns.STEAL_ATTEMPTS
        steals_key = StatColumns.STOLEN_BASES
        stats[attempts_key] = stats.get(attempts_key, 0) + 1
        if success:
            stats[steals_key] = stats.get(steals_key, 0) + 1

    def execute_steal(self):
        """盗塁処理を実行し結果ペイロードを返す"""
        if not self.can_steal():
            message = "Steal not allowed (no open base to advance)."
            self.record_play(GameResults.STEAL_NOT_ALLOWED, message)
            return {
                "success": False,
                "result": GameResults.STEAL_NOT_ALLOWED,
                "message": message,
                "outs_recorded": 0,
            }

        first_runner = self._bases[0]
        second_runner = self._bases[1]
        third_runner = self._bases[2]

        outs_recorded = 0
        result_key = GameResults.STOLEN_BASE
        message = ""
        success = True

        if first_runner and second_runner and not third_runner:
            first_prob = self._steal_success_probability(first_runner)
            second_prob = self._steal_success_probability(second_runner)
            first_success = random.random() < first_prob
            second_success = random.random() < second_prob

            self._register_steal_attempt(first_runner, first_success)
            self._register_steal_attempt(second_runner, second_success)

            if first_success and second_success:
                self._bases[0] = None
                self._bases[1] = first_runner
                self._bases[2] = second_runner
                message = (
                    f"ダブルスチール成功！{first_runner.name}が二塁へ、{second_runner.name}が三塁へ。"
                )
            else:
                success = False
                result_key = GameResults.CAUGHT_STEALING
                outs_recorded = 1

                if first_success and not second_success:
                    # 二塁走者がアウト、走者一塁は成功して二塁へ
                    self._bases[0] = None
                    self._bases[1] = first_runner
                    self._bases[2] = None
                    out_runner = second_runner
                    safe_runner = first_runner
                    message = (
                        f"ダブルスチール失敗。{out_runner.name}が三塁でアウト、{safe_runner.name}は二塁を奪取。"
                    )
                elif not first_success and second_success:
                    # 一塁走者がアウト、二塁走者は三塁へ
                    self._bases[0] = None
                    self._bases[1] = None
                    self._bases[2] = second_runner
                    out_runner = first_runner
                    safe_runner = second_runner
                    message = (
                        f"ダブルスチール失敗。{out_runner.name}が二塁でアウト、{safe_runner.name}は三塁へ進塁。"
                    )
                else:
                    # 両者失敗だが仕様上アウトは1人のみ
                    out_runner = random.choice([first_runner, second_runner])
                    if out_runner is first_runner:
                        self._bases[0] = None
                        self._bases[1] = second_runner
                        self._bases[2] = None
                        message = (
                            f"ダブルスチール失敗。{first_runner.name}が二塁でアウト、{second_runner.name}は二塁に戻った。"
                        )
                    else:
                        self._bases[0] = first_runner
                        self._bases[1] = None
                        self._bases[2] = None
                        message = (
                            f"ダブルスチール失敗。{second_runner.name}が三塁でアウト、{first_runner.name}は一塁に戻った。"
                        )
                self.add_out(advance_lineup=False)
        elif second_runner and not third_runner:
            probability = self._steal_success_probability(second_runner)
            steal_success = random.random() < probability
            self._register_steal_attempt(second_runner, steal_success)

            if steal_success:
                self._bases[1] = None
                self._bases[2] = second_runner
                message = f"{second_runner.name}が三塁への盗塁に成功！"
            else:
                self._bases[1] = None
                self._bases[2] = None
                message = f"{second_runner.name}の三塁盗塁は失敗、アウト。"
                result_key = GameResults.CAUGHT_STEALING
                success = False
                outs_recorded = 1
                self.add_out(advance_lineup=False)
        elif first_runner and not second_runner:
            probability = self._steal_success_probability(first_runner)
            steal_success = random.random() < probability
            self._register_steal_attempt(first_runner, steal_success)

            if steal_success:
                self._bases[0] = None
                self._bases[1] = first_runner
                message = f"{first_runner.name}が二塁への盗塁に成功！"
            else:
                self._bases[0] = None
                self._bases[1] = None
                message = f"{first_runner.name}の二塁盗塁は失敗、アウト。"
                result_key = GameResults.CAUGHT_STEALING
                success = False
                outs_recorded = 1
                self.add_out(advance_lineup=False)
        else:
            message = "盗塁はできません（条件不一致）。"
            success = False
            result_key = GameResults.STEAL_NOT_ALLOWED

        self.record_play(result_key, message)
        return {
            "success": success,
            "result": result_key,
            "message": message,
            "outs_recorded": outs_recorded,
        }

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
        message = self._outcome_handler.apply(result, batter)
        self.record_play(result, message)
        return message
