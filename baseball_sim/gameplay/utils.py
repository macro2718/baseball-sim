"""Utility helpers for in-game calculations such as bunting logic."""

import random
from typing import Any, Dict, List, Tuple

from baseball_sim.config import BuntConstants, GameResults, StatColumns
from baseball_sim.infrastructure.logging_utils import logger

class BuntCalculator:
    """バント計算クラス"""
    
    @staticmethod
    def calculate_bunt_success_probability(batter, pitcher) -> float:
        """バント成功確率を計算（バッターとピッチャーのパラメータから）"""
        # 基本成功率（70%程度）
        base_success_rate = 0.70
        
        # バッターによる影響
        # 三振率が低いほど、バットコントロールが良い
        batter_skill_factor = 1.0 + (25.0 - batter.k_pct) / 100.0  # 標準的な三振率25%を基準
        
        # ピッチャーによる影響  
        # 投手の制球力（四球率が低いほど制球が良い）
        pitcher_control_factor = 1.0 + (8.5 - pitcher.bb_pct) / 50.0  # 標準的な四球率8.5%を基準
        
        # 投手のスタミナ影響（疲れているほどバントしやすい）
        stamina_factor = 1
        
        # 最終成功確率を計算
        success_rate = base_success_rate * batter_skill_factor * pitcher_control_factor * stamina_factor
        
        # 確率を0.1～0.95の範囲に制限
        return max(0.1, min(0.95, success_rate))
    
    @staticmethod
    def calculate_bunt_result(batter, pitcher, game_state) -> str:
        """バント実行時の結果を計算"""
        success_probability = BuntCalculator.calculate_bunt_success_probability(batter, pitcher)
        
        if random.random() < success_probability:
            # バント成功時の処理
            return BuntCalculator._determine_successful_bunt_result(game_state)
        else:
            # バント失敗時の処理
            return BuntCalculator._determine_failed_bunt_result()
    
    @staticmethod
    def _determine_successful_bunt_result(game_state) -> str:
        """バント成功時の詳細結果を決定"""
        # ランナーの状況を確認
        runners_on_base = sum(1 for base in game_state.bases if base is not None)
        
        if runners_on_base == 0:
            # ランナーなしの場合、単純な内野安打またはアウト
            if random.random() < 0.3:  # 30%で内野安打
                return "bunt_single"
            else:
                return "bunt_out"
        else:
            # ランナーありの場合、送りバント成功
            return "sacrifice_bunt"
    
    @staticmethod
    def _determine_failed_bunt_result() -> str:
        """バント失敗時の結果を決定"""
        fail_roll = random.random()
        
        if fail_roll < 0.67:  # 67%でアウト
            return "bunt_out"
        else:  # 33%でバントしたままアウト（送りバント失敗）
            return "bunt_failed"


class BuntProcessor:
    """バント処理を一元化するクラス"""
    
    def __init__(self, game_state):
        self.game_state = game_state
    
    def execute(self, batter, pitcher):
        """バント実行のメインメソッド"""
        # バント実行可能性の事前チェック
        if not self.game_state.can_bunt():
            return "Cannot bunt (No runners on base or 2 outs)"
        
        # バント結果を計算
        bunt_result = BuntCalculator.calculate_bunt_result(batter, pitcher, self.game_state)
        
        # バント結果を試合状況に適用
        return self.apply_result(bunt_result, batter)
    
    def apply_result(self, bunt_result, batter):
        """バント結果を試合状況に適用する"""
        # 基本統計の更新（全バント結果共通）
        batter.stats["PA"] += 1
        
        # バント結果別の処理
        if bunt_result == "bunt_single":
            return self._handle_bunt_single(batter)
        elif bunt_result == "sacrifice_bunt":
            return self._handle_sacrifice_bunt(batter)
        elif bunt_result == "bunt_out":
            return self._handle_bunt_out(batter)
        elif bunt_result == "bunt_failed":
            return self._handle_bunt_failed(batter)
        
        return "Unknown bunt result"
    
    def _handle_bunt_single(self, batter):
        """バント内野安打の処理"""
        # 統計更新
        batter.stats["AB"] += 1
        batter.stats["1B"] += 1
        self.game_state.fielding_team.current_pitcher.pitching_stats["H"] += 1
        
        # ランナー進塁処理
        runs_scored = self._advance_runners("bunt_single", batter)
        
        # 得点があれば加算
        if runs_scored > 0:
            self.game_state._add_runs(runs_scored, batter)
            return f"Bunt hit! {runs_scored} runs scored!"
        
        return "Bunt hit!"
    
    def _handle_sacrifice_bunt(self, batter):
        """送りバント成功の処理"""
        # 統計更新
        batter.stats["AB"] += 1
        batter.stats["SAC"] = batter.stats.get("SAC", 0) + 1
        self.game_state.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
        
        # 進塁前の塁状況を記録
        runners_before = [
            base is not None for base in self.game_state.bases
        ]
        
        # ランナー進塁処理
        runs_scored = self._advance_runners("sacrifice_bunt")
        
        # 進塁後の塁状況を確認
        runners_after = [
            base is not None for base in self.game_state.bases
        ]
        
        # 打者アウト
        self.game_state.add_out()
        
        # 実際に進塁が発生したかチェック
        actual_advancement = self._check_actual_advancement(runners_before, runners_after)
        
        # 進塁結果をメッセージで表示
        advance_message = self._create_advance_message(runners_before, runners_after)
        
        # 得点があれば加算
        if runs_scored > 0:
            self.game_state._add_runs(runs_scored, batter)
            return f"Sacrifice bunt successful! {runs_scored} runs scored! {advance_message}"
        
        # 実際に進塁が発生した場合のみ「successful」と表示
        if actual_advancement or runs_scored > 0:
            return f"Sacrifice bunt successful! {advance_message}"
        else:
            return f"Sacrifice bunt attempted, but no advancement occurred. {advance_message}"
    
    def _handle_bunt_out(self, batter):
        """バントアウトの処理"""
        # 統計更新
        batter.stats["AB"] += 1
        self.game_state.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
        
        # バントアウト時のランナー処理
        runner_out_message = self._handle_bunt_out_scenarios()
        
        # 打者アウト
        self.game_state.add_out()
        
        if runner_out_message:
            return f"Bunt out! {runner_out_message}"
        
        return "Bunt out"
    
    def _handle_bunt_failed(self, batter):
        """バント失敗の処理"""
        # 統計更新
        batter.stats["AB"] += 1
        self.game_state.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
        
        # バント失敗時の追加アウト判定
        additional_outs = self._handle_bunt_failure()
        
        # 打者アウト
        self.game_state.add_out()
        
        if additional_outs > 0:
            return f"Bunt failed! {additional_outs} additional runner(s) out!"
        
        return "Bunt failed!"
    
    def _advance_runners(self, bunt_type, batter=None):
        """統一的なランナー進塁処理"""
        runs_scored = 0
        new_bases = [None] * 3  # BASES_COUNT
        
        if bunt_type == "bunt_single":
            # バント内野安打時の処理
            runs_scored = self._process_bunt_single_advances(new_bases, batter)
        elif bunt_type == "sacrifice_bunt":
            # 送りバント時の処理
            runs_scored = self._process_sacrifice_bunt_advances(new_bases)
        
        # 塁状況を更新
        self.game_state.bases = new_bases
        return runs_scored
    
    def _process_bunt_single_advances(self, new_bases, batter):
        """バント内野安打時のランナー進塁処理"""
        runs_scored = 0
        
        # 三塁走者：必ずホームイン（内野安打のため）
        if self.game_state.bases[2] is not None:
            runs_scored += 1
        
        # 二塁走者：走力によってホームインまたは三塁進塁
        if self.game_state.bases[1] is not None:
            runner_speed = self._get_runner_speed(1)
            if runner_speed < BuntConstants.FAST_RUNNER_SPEED:
                if random.random() < BuntConstants.HOME_IN_PROBABILITY_FAST_RUNNER:
                    runs_scored += 1  # ホームイン
                else:
                    new_bases[2] = self.game_state.bases[1]  # 三塁進塁
            else:
                new_bases[2] = self.game_state.bases[1]  # 三塁進塁
        
        # 一塁走者：走力によって二塁または三塁進塁
        if self.game_state.bases[0] is not None:
            runner_speed = self._get_runner_speed(0)
            if runner_speed < BuntConstants.VERY_FAST_RUNNER_SPEED:
                if new_bases[2] is None and random.random() < BuntConstants.TRIPLE_ADVANCE_PROBABILITY:
                    new_bases[2] = self.game_state.bases[0]  # 三塁進塁
                else:
                    new_bases[1] = self.game_state.bases[0]  # 二塁進塁
            else:
                new_bases[1] = self.game_state.bases[0]  # 二塁進塁
        
        # 打者は一塁へ進塁（内野安打）
        new_bases[0] = batter
        
        return runs_scored
    
    def _process_sacrifice_bunt_advances(self, new_bases):
        """送りバント時のランナー進塁処理"""
        runs_scored = 0
        
        # スクイズプレーの可能性を判定
        if self._should_attempt_squeeze_play():
            runs_scored += self._execute_squeeze_play(new_bases)
        else:
            runs_scored += self._execute_normal_sacrifice_bunt(new_bases)
        
        return runs_scored
    
    def _should_attempt_squeeze_play(self):
        """スクイズプレーを実行すべきかを判定"""
        return (self.game_state.bases[2] and 
                self.game_state.outs < 2 and 
                random.random() < BuntConstants.SQUEEZE_PROBABILITY)
    
    def _execute_squeeze_play(self, new_bases):
        """スクイズプレー実行"""
        runs_scored = 0
        
        # スクイズプレーの成功判定
        if self._can_runner_advance_safely(2, -1, "squeeze"):
            runs_scored += 1  # 三塁走者がホームイン
        else:
            # スクイズ失敗時の処理
            if random.random() < BuntConstants.RUNNER_OUT_ON_SQUEEZE_FAIL:
                self.game_state.add_out()  # ランナーアウト
            else:
                new_bases[2] = self.game_state.bases[2]  # 三塁に留まる
        
        # 他のランナーの処理
        self._advance_other_runners(new_bases)
        
        return runs_scored
    
    def _execute_normal_sacrifice_bunt(self, new_bases):
        """通常の送りバント処理
        
        進塁は後ろの塁から順番に処理する（三塁→二塁→一塁）
        各塁の進塁成功/失敗を個別に判定し、実際の進塁結果を正確に反映する
        """
        runs_scored = 0
        
        # 三塁走者の処理
        if self.game_state.bases[2] is not None:
            if self.game_state.outs < 2:
                # ホームイン判定を確率的に行う
                if random.random() < BuntConstants.HOME_IN_PROBABILITY_FROM_THIRD:
                    runs_scored += 1  # ホームイン
                    # 三塁は空になる（new_basesでは設定しない）
                else:
                    new_bases[2] = self.game_state.bases[2]  # 三塁に留まる
            else:
                new_bases[2] = self.game_state.bases[2]  # 2アウトでは三塁に留まる
        
        # 二塁走者の処理（三塁の状況を考慮）
        if self.game_state.bases[1] is not None:
            can_advance = self._can_runner_advance_safely(1, 2)
            if can_advance and new_bases[2] is None:  # 三塁が空いている
                new_bases[2] = self.game_state.bases[1]  # 三塁に進塁
                # 二塁は空になる（new_basesでは設定しない）
            else:
                # 進塁できない場合、二塁に留まる
                new_bases[1] = self.game_state.bases[1]
        
        # 一塁走者の処理（二塁の状況を考慮）
        if self.game_state.bases[0] is not None:
            can_advance = self._can_runner_advance_safely(0, 1)
            if can_advance and new_bases[1] is None:  # 二塁が空いている
                new_bases[1] = self.game_state.bases[0]  # 二塁に進塁
                # 一塁は空になる（new_basesでは設定しない）
            else:
                # 進塁できない場合、一塁に留まる
                new_bases[0] = self.game_state.bases[0]
        
        return runs_scored
    
    def _advance_other_runners(self, new_bases):
        """スクイズプレー後の他ランナーの進塁処理"""
        # 二塁走者の処理
        if self.game_state.bases[1] is not None:
            if self._can_runner_advance_safely(1, 2):
                new_bases[2] = self.game_state.bases[1]  # 三塁に進塁
            else:
                new_bases[1] = self.game_state.bases[1]  # 二塁に留まる
        
        # 一塁走者の処理
        if self.game_state.bases[0] is not None:
            if self._can_runner_advance_safely(0, 1):
                if new_bases[1] is None:
                    new_bases[1] = self.game_state.bases[0]  # 二塁に進塁
                else:
                    new_bases[0] = self.game_state.bases[0]  # 一塁に留まる
            else:
                new_bases[0] = self.game_state.bases[0]  # 一塁に留まる
    
    def _can_runner_advance_safely(self, from_base, to_base, bunt_type="sacrifice"):
        """ランナーが安全に進塁できるかどうかを判定"""
        if self.game_state.bases[from_base] is None:
            return False
        
        runner_speed = self._get_runner_speed(from_base)
        base_success_rate = self._get_base_success_rate_by_bunt_type(bunt_type)
        
        # 走力による成功率調整
        speed_factor = BuntConstants.STANDARD_RUNNER_SPEED / runner_speed
        speed_adjusted_rate = base_success_rate * speed_factor
        
        # アウトカウントによる成功率調整
        if self.game_state.outs >= 2:
            speed_adjusted_rate *= BuntConstants.TWO_OUT_PENALTY
        
        # 確率を適切な範囲に制限
        final_rate = max(BuntConstants.MIN_ADVANCE_PROBABILITY, 
                        min(BuntConstants.MAX_ADVANCE_PROBABILITY, speed_adjusted_rate))
        
        return random.random() < final_rate
    
    def _get_base_success_rate_by_bunt_type(self, bunt_type):
        """バント種類による基本成功率を取得"""
        if bunt_type == "sacrifice":
            return BuntConstants.SACRIFICE_BUNT_SUCCESS_RATE
        elif bunt_type == "squeeze":
            return BuntConstants.SQUEEZE_PLAY_SUCCESS_RATE
        else:
            return BuntConstants.GENERAL_BUNT_SUCCESS_RATE
    
    def _get_runner_speed(self, base_index):
        """指定された塁のランナーの走力を取得"""
        if (self.game_state.bases[base_index] is not None and 
            hasattr(self.game_state.bases[base_index], 'speed')):
            return self.game_state.bases[base_index].speed
        return BuntConstants.STANDARD_RUNNER_SPEED
    
    def _handle_bunt_out_scenarios(self):
        """バントアウト時のランナー処理"""
        runners_out = 0
        out_messages = []
        
        # 一塁ランナーがいる場合、フォースアウトの可能性
        if self.game_state.bases[0] is not None:
            force_out_chance = 0.15  # 15%でフォースアウトも発生
            if random.random() < force_out_chance:
                self.game_state.bases[0] = None
                self.game_state.add_out()
                runners_out += 1
                out_messages.append("1st base runner also out")
        
        # 進塁しようとしたランナーがアウトになる可能性
        advance_attempt_bases = []
        if self.game_state.bases[1] is not None and random.random() < 0.1:
            advance_attempt_bases.append(1)
        if self.game_state.bases[2] is not None and random.random() < 0.05:
            advance_attempt_bases.append(2)
        
        for base in advance_attempt_bases:
            if random.random() < 0.6:  # 60%で進塁失敗してアウト
                self.game_state.bases[base] = None
                self.game_state.add_out()
                runners_out += 1
                base_name = ["", "2nd", "3rd"][base]
                out_messages.append(f"{base_name} base runner caught advancing")
        
        if out_messages:
            return ", ".join(out_messages)
        return None
    
    def _handle_bunt_failure(self):
        """バント失敗時の追加アウト処理"""
        additional_outs = 0
        
        # 一塁ランナーがいる場合のフォースアウト
        if self.game_state.bases[0] is not None:
            force_out_probability = 0.4  # 40%でフォースアウト
            if random.random() < force_out_probability:
                self.game_state.bases[0] = None
                self.game_state.add_out()
                additional_outs += 1
                
                # ダブルプレーの可能性（二塁ランナーもいる場合）
                if self.game_state.bases[1] is not None and self.game_state.outs < 2:
                    dp_probability = 0.3  # 30%でダブルプレー
                    if random.random() < dp_probability:
                        self.game_state.bases[1] = None
                        self.game_state.add_out()
                        additional_outs += 1
        
        return additional_outs
    
    def _check_actual_advancement(self, runners_before, runners_after):
        """実際に進塁が発生したかをチェック
        
        Args:
            runners_before: 進塁前のランナー状況 [bool, bool, bool]
            runners_after: 進塁後のランナー状況 [bool, bool, bool]
        
        Returns:
            bool: 実際に進塁が発生した場合True
        """
        # より正確な進塁判定を行う
        # 単純にランナーの有無だけでなく、全体的な変化をチェック
        
        # ケース1: 一塁ランナーが二塁に進塁
        if runners_before[0] and not runners_after[0] and runners_after[1]:
            return True
        
        # ケース2: 二塁ランナーが三塁に進塁  
        if runners_before[1] and not runners_after[1] and runners_after[2]:
            return True
        
        # ケース3: 三塁ランナーがホームイン（得点による判定は別途行う）
        if runners_before[2] and not runners_after[2]:
            return True
        
        # ケース4: 複数ランナーの同時進塁
        # 一塁→二塁、二塁→三塁の同時進塁
        if (runners_before[0] and runners_before[1] and 
            not runners_after[0] and runners_after[1] and runners_after[2]):
            return True
        
        # ケース5: 満塁からの進塁（誰かがホームイン）
        if all(runners_before) and sum(runners_after) < sum(runners_before):
            return True
        
        return False

    def _create_advance_message(self, runners_before, runners_after):
        """進塁結果のメッセージを作成
        
        進塁前後のランナー状況を比較して、実際の移動を追跡する
        """
        advances = []
        stayed = []
        
        # 各ランナーを個別にトラッキングする必要があるが、
        # オブジェクト参照で追跡が難しいため、論理的な推論を使用
        
        # ケース分析による進塁判定
        total_before = sum(runners_before)
        total_after = sum(runners_after)
        
        # 一塁ランナーの分析
        if runners_before[0]:  # 元々一塁にランナーがいた
            if not runners_after[0]:  # 一塁が空になった
                advances.append("1st to 2nd")
            else:  # 一塁に留まった
                stayed.append("1st")
        
        # 二塁ランナーの分析
        if runners_before[1]:  # 元々二塁にランナーがいた
            # 二塁走者の進塁を判定
            # もし三塁にランナーが新たに現れ、かつ二塁が空になっていない場合は
            # 一塁ランナーが二塁に入った可能性が高い
            if runners_after[2] and not runners_before[2]:  # 三塁に新しくランナーが現れた
                if runners_after[1] and runners_before[0]:  # 二塁にもランナーがいて、元々一塁にもいた
                    advances.append("2nd to 3rd")  # 二塁→三塁の進塁
                    # 二塁に「留まった」ように見えるのは一塁ランナーが入ったため
                elif not runners_after[1]:  # 二塁が空になった
                    advances.append("2nd to 3rd")
            elif runners_after[1]:  # 二塁に留まった
                stayed.append("2nd")
        
        # 三塁ランナーの分析
        if runners_before[2]:  # 元々三塁にランナーがいた
            if runners_after[2]:  # 三塁に留まった
                stayed.append("3rd")
            # ホームインした場合は得点メッセージで表示される
        
        # メッセージの構築
        messages = []
        if advances:
            messages.append("Advanced: " + ", ".join(advances))
        if stayed:
            messages.append("Stayed: " + ", ".join(stayed))
        
        if messages:
            return "(" + "; ".join(messages) + ")"
        elif any(runners_before):
            return "(No advancement)"
        else:
            return "(No base runners)"


class RunnerEngine:
    """通常打席結果のランナー進塁を担当するユーティリティ。

    注意: 既存の game.py の挙動を保つことを最優先にし、
    確率や計算式は現状のロジックをそのまま反映する。
    """

    def __init__(self, game_state):
        self.game_state = game_state

    # ---- 四球 ----
    def apply_walk(self, batter) -> int:
        """四球時のランナー進塁処理（得点数を返す）"""
        runs = 0
        bases = self.game_state.bases
        base_count = len(bases)

        # 進塁が強制される走者を判定（手前の塁に走者がいる場合のみ）
        forced = [False] * base_count
        if base_count > 0:
            forced[0] = bases[0] is not None
        for i in range(1, base_count):
            forced[i] = bases[i] is not None and forced[i - 1]

        # 三塁側から順に進塁させる
        for i in range(base_count - 1, -1, -1):
            if not forced[i]:
                continue

            runner = bases[i]
            if i == base_count - 1:
                runs += 1
            else:
                bases[i + 1] = runner
            bases[i] = None

        # 打者は一塁へ
        if base_count > 0:
            bases[0] = batter
        return runs

    # ---- 単打 ----
    def apply_single(self, batter) -> int:
        """単打時のランナー進塁処理（得点数を返す）"""
        runs = 0

        # 三塁走者の処理（状況に応じてホーム突入を判断）
        third_runner = self.game_state.bases[2]
        if third_runner is not None:
            runner_speed = getattr(third_runner, "speed", 4.3)
            forced_home = self.game_state.bases[1] is not None

            # 強制進塁が無い場合のみ「突入しない」選択肢が生じる
            attempt_probability = 0.85 * (4.3 / runner_speed)
            attempt_probability = max(0.5, min(0.95, attempt_probability))

            if forced_home or random.random() < attempt_probability:
                success_probability = 0.75 * (4.3 / runner_speed)
                success_probability = max(0.4, min(0.98, success_probability))

                if random.random() < success_probability:
                    runs += 1
                    self.game_state.bases[2] = None
                else:
                    if forced_home:
                        self.game_state.bases[2] = None
                        self.game_state.add_out()
                    else:
                        out_probability = 0.45 * (runner_speed / 4.3)
                        out_probability = max(0.25, min(0.7, out_probability))
                        if random.random() < out_probability:
                            self.game_state.bases[2] = None
                            self.game_state.add_out()
            else:
                # 突入を見送った場合は三塁に留まる
                self.game_state.bases[2] = third_runner

        # 二塁走者の処理（既存ロジック踏襲。走者でなく batter.speed を参照していた挙動を保持）
        if self.game_state.bases[1] is not None:
            run_probability = 0.65
            run_probability *= (4.3 / batter.speed)
            if random.random() < run_probability:
                runs += 1
            else:
                self.game_state.bases[2] = self.game_state.bases[1]
            self.game_state.bases[1] = None

        # 一塁走者の処理
        if self.game_state.bases[0] is not None:
            advance_probability = 0.1
            advance_probability *= (4.3 / batter.speed)
            if self.game_state.bases[2] is None and random.random() < advance_probability:
                self.game_state.bases[2] = self.game_state.bases[0]
            else:
                self.game_state.bases[1] = self.game_state.bases[0]
            self.game_state.bases[0] = None

        # 打者は一塁へ
        self.game_state.bases[0] = batter

        return runs

    # ---- 二塁打 ----
    def apply_double(self, batter) -> int:
        """二塁打時のランナー進塁処理（得点数を返す）"""
        runs = 0

        # 三塁と二塁の走者はホームイン
        if self.game_state.bases[2] is not None:
            runs += 1
            self.game_state.bases[2] = None

        if self.game_state.bases[1] is not None:
            runs += 1
            self.game_state.bases[1] = None

        # 一塁走者の処理
        if self.game_state.bases[0] is not None:
            run_probability = 0.2 * (4.3 / batter.speed)
            if random.random() < run_probability:
                runs += 1
            else:
                self.game_state.bases[2] = self.game_state.bases[0]
            self.game_state.bases[0] = None

        # 打者は二塁へ
        self.game_state.bases[1] = batter

        return runs

    # ---- 三塁打 ----
    def apply_triple(self, batter) -> int:
        """三塁打時のランナー進塁処理（得点数を返す）"""
        runs = 0
        for i in range(3):
            if self.game_state.bases[i] is not None:
                runs += 1
                self.game_state.bases[i] = None
        # 打者は三塁へ
        self.game_state.bases[2] = batter
        return runs

    # ---- 本塁打 ----
    def apply_home_run(self, batter) -> int:
        """本塁打時のランナー進塁処理（得点数を返す）"""
        runs = 1  # 打者の分
        for i in range(3):
            if self.game_state.bases[i] is not None:
                runs += 1
                self.game_state.bases[i] = None
        return runs

    # ---- ゴロアウト ----
    def apply_groundout(self, batter) -> Tuple[int, str]:
        """ゴロアウト時の進塁・アウト処理。得点数とメッセージを返す。

        前提: 呼び出し元で投手のIPを1/3加算、打者ABを加算済み。
        ここではアウトカウントの増減、追加のIP（併殺時）を処理する。
        """
        if self._is_double_play_possible():
            return self._handle_double_play_situation(batter)
        else:
            return self._handle_regular_groundout(batter)

    def _is_double_play_possible(self) -> bool:
        return (self.game_state.bases[0] is not None and self.game_state.outs < 2)

    def _handle_double_play_situation(self, batter) -> Tuple[int, str]:
        runner_situation = self._get_runner_situation()
        if self.game_state.outs == 0:
            return self._handle_dp_with_zero_outs(batter, runner_situation)
        else:  # outs == 1
            return self._handle_dp_with_one_out(batter, runner_situation)

    def _get_runner_situation(self) -> str:
        second = self.game_state.bases[1] is not None
        third = self.game_state.bases[2] is not None
        if not second and not third:
            return "first_only"
        elif second and not third:
            return "first_second"
        elif not second and third:
            return "first_third"
        else:
            return "bases_loaded"

    def _handle_dp_with_zero_outs(self, batter, runner_situation) -> Tuple[int, str]:
        dp_probability = 0.4 * (batter.speed / 4.3)
        if random.random() < dp_probability:
            # ダブルプレー成功
            runs_scored = self._execute_double_play(runner_situation)
            # 一塁走者アウト、打者アウト（合計2アウト）
            self.game_state.bases[0] = None
            self.game_state.add_out()  # 一塁走者
            self.game_state.add_out()  # 打者
            # 追加のアウト分のIPを加算
            self.game_state.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
            if runs_scored > 0:
                return runs_scored, f"Double play! {runs_scored} run(s) scored!"
            return 0, "Double play!"
        else:
            # ダブルプレー失敗 - フォースアウトのみ
            return self._handle_force_out_only(batter, runner_situation)

    def _handle_dp_with_one_out(self, batter, runner_situation) -> Tuple[int, str]:
        dp_probability = 0.35 * (batter.speed / 4.3)
        if random.random() < dp_probability:
            # ダブルプレー成功 - イニング終了（点は入らない）
            runs_scored = 0
            self.game_state.bases[0] = None
            self.game_state.add_out()  # 一塁走者 (2アウト目)
            self.game_state.add_out()  # 打者 (3アウト目)
            self.game_state.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
            return 0, "Inning-ending double play!"
        else:
            return self._handle_force_out_only(batter, runner_situation)

    def _execute_double_play(self, runner_situation) -> int:
        runs_scored = 0
        # 0アウト時の処理（元のロジックを踏襲）
        if runner_situation == "first_only":
            pass
        elif runner_situation == "first_second":
            if self.game_state.bases[1] is not None:
                self.game_state.bases[2] = self.game_state.bases[1]
                self.game_state.bases[1] = None
        elif runner_situation == "first_third":
            if self.game_state.bases[2] is not None:
                scoring_chance = 0.3
                if random.random() < scoring_chance:
                    runs_scored += 1
                self.game_state.bases[2] = None
        else:  # bases_loaded
            if self.game_state.bases[2] is not None:
                scoring_chance = 0.4
                if random.random() < scoring_chance:
                    runs_scored += 1
                self.game_state.bases[2] = None
            if self.game_state.bases[1] is not None:
                self.game_state.bases[2] = self.game_state.bases[1]
                self.game_state.bases[1] = None
        return runs_scored

    def _handle_force_out_only(self, batter, runner_situation) -> Tuple[int, str]:
        runs_scored = 0
        # 三塁走者の得点チェック
        if runner_situation in ["first_third", "bases_loaded"]:
            if self.game_state.bases[2] is not None:
                scoring_chance = 0.5
                if random.random() < scoring_chance:
                    runs_scored += 1
                    self.game_state.bases[2] = None
        # 走者の進塁
        if runner_situation == "first_second":
            if self.game_state.bases[1] is not None:
                self.game_state.bases[2] = self.game_state.bases[1]
                self.game_state.bases[1] = None
        elif runner_situation == "bases_loaded":
            if self.game_state.bases[1] is not None and self.game_state.bases[2] is None:
                self.game_state.bases[2] = self.game_state.bases[1]
                self.game_state.bases[1] = None
        # 打者は一塁へ（フォースアウトで元一塁走者がアウト）
        self.game_state.bases[0] = batter
        self.game_state.add_out()
        if runs_scored > 0:
            return runs_scored, f"Groundout, force at second. {runs_scored} run(s) scored!"
        return 0, "Groundout, force at second."

    def _handle_regular_groundout(self, batter) -> Tuple[int, str]:
        runs_scored = 0
        if self.game_state.bases[2] is not None and self.game_state.outs < 2:
            scoring_probability = 0.4
            if random.random() < scoring_probability:
                runs_scored += 1
                self.game_state.bases[2] = None
        if self.game_state.bases[1] is not None and self.game_state.bases[2] is None:
            self.game_state.bases[2] = self.game_state.bases[1]
            self.game_state.bases[1] = None
        if self.game_state.bases[0] is not None and self.game_state.bases[1] is None:
            self.game_state.bases[1] = self.game_state.bases[0]
            self.game_state.bases[0] = None
        self.game_state.add_out()  # 打者アウト
        if runs_scored > 0:
            return runs_scored, f"Groundout. {runs_scored} run(s) scored!"
        return 0, "Groundout."

    # ---- フライアウト ----
    def apply_flyout(self, batter) -> int:
        """フライアウト時の進塁・得点処理（得点数を返す）。

        前提: 呼び出し元で投手のIPを1/3加算、打者ABを加算し、最後に打者アウトを付与する。
        ここでは犠牲フライの判定と走者のタッチアップ処理を行う。
        """
        runs = 0
        if self.game_state.outs < 2:
            anticipated_outs = self.game_state.outs + 1  # 打者アウト分

            # 三塁走者の処理（犠牲フライ or タッチアップ失敗）
            third_runner = self.game_state.bases[2]
            if third_runner is not None:
                sac_fly_probability = 0.6 * (batter.hard_pct / 35)
                sac_fly_probability = max(0.0, min(0.95, sac_fly_probability))

                if random.random() < sac_fly_probability:
                    runs += 1
                    self.game_state.bases[2] = None
                else:
                    # タッチアップ失敗でアウト
                    self.game_state.bases[2] = None
                    self.game_state.add_out()
                    anticipated_outs += 1

            # 既にイニング終了が確定する場合は他走者の処理不要
            if anticipated_outs >= 3:
                return runs

            depth_factor = max(0.2, min(1.5, batter.hard_pct / 35))

            # 二塁走者のタッチアップ進塁判定
            second_runner = self.game_state.bases[1]
            if second_runner is not None and self.game_state.bases[2] is None:
                second_speed = getattr(second_runner, "speed", 4.3) or 4.3
                second_probability = 0.45 * depth_factor * (4.3 / second_speed)
                second_probability = max(0.0, min(0.8, second_probability))
                if random.random() < second_probability:
                    self.game_state.bases[2] = second_runner
                    self.game_state.bases[1] = None

            # 一塁走者のタッチアップ進塁判定
            first_runner = self.game_state.bases[0]
            if first_runner is not None and self.game_state.bases[1] is None:
                first_speed = getattr(first_runner, "speed", 4.3) or 4.3
                first_probability = 0.3 * depth_factor * (4.3 / first_speed)
                first_probability = max(0.0, min(0.65, first_probability))
                if random.random() < first_probability:
                    self.game_state.bases[1] = first_runner
                    self.game_state.bases[0] = None

        return runs
