"""
野球ゲームの試合進行を管理するメインモジュール
バッティング予測、守備処理、ゲーム状態管理を担当
"""
import random
import numpy as np
import joblib
import torch

# プロジェクト設定を使用
from project_config import setup_project_environment
setup_project_environment()

from prediction_models.prediction import predict_auto, Net
from constants import (
    INNINGS_PER_GAME, OUTS_PER_INNING, BASES_COUNT, MAX_EXTRA_INNINGS,
    GameResults, BuntConstants
)
from path_utils import path_manager
from config import config

try:
    from error_handling import GameStateError, log_error, logger
except ImportError:
    # フォールバック: エラーハンドリングモジュールが利用できない場合
    def log_error(func):
        return func
    
    class logger:
        @staticmethod
        def info(msg): 
            print(f"INFO: {msg}")
        @staticmethod
        def warning(msg): 
            print(f"WARNING: {msg}")
        @staticmethod
        def error(msg): 
            print(f"ERROR: {msg}")
    
    GameStateError = Exception

class GameState:
    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team
        self.inning = 1
        self.is_top_inning = True  # 表（アウェイチーム攻撃）ならTrue、裏（ホームチーム攻撃）ならFalse
        self.outs = 0
        self.bases = [None] * BASES_COUNT  # 一塁、二塁、三塁の選手オブジェクト
        self.home_score = 0
        self.away_score = 0
        self.batting_team = away_team  # アウェイチームが先攻
        self.fielding_team = home_team
        self.game_ended = False  # 試合終了フラグを追加
        self.defensive_position_error = False  # 守備位置に問題があるかどうかのフラグ
        self.defensive_error_messages = []  # 守備位置のエラーメッセージ
        
        # イニングごとの得点記録 [アウェイチーム, ホームチーム]
        self.inning_scores = [[], []]
        self._initialize_first_inning()
        self.model, self.model_type = self._load_batting_model()

    def _initialize_first_inning(self):
        """最初のイニングの得点枠を初期化"""
        self.inning_scores[0].append(0)  # アウェイチーム 1回表
        self.inning_scores[1].append(0)  # ホームチーム 1回裏

    @log_error
    def _load_batting_model(self):
        """打撃モデルを読み込み"""
        try:
            model_type = config.get('simulation.prediction_model_type', 'linear')
            
            if model_type == 'linear':
                model_path = path_manager.get_batting_model_path()
                if not path_manager.file_exists(model_path):
                    logger.warning(f"Linear batting model not found at {model_path}, using default prediction")
                    return None, 'linear'
                
                model_info = joblib.load(model_path)
                logger.info("Linear batting model loaded successfully")
                return model_info['model'], 'linear'
                
            elif model_type == 'nn':
                model_path = path_manager.get_nn_model_path()
                if not path_manager.file_exists(model_path):
                    logger.warning(f"NN model not found at {model_path}, falling back to linear model")
                    # フォールバック：線形モデルを試す
                    linear_path = path_manager.get_batting_model_path()
                    if path_manager.file_exists(linear_path):
                        model_info = joblib.load(linear_path)
                        logger.info("Fallback to linear batting model")
                        return model_info['model'], 'linear'
                    return None, 'linear'
                
                # NNモデルを読み込み
                model = Net(input_dim=4, output_dim=5)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                logger.info("NN batting model loaded successfully")
                return model, 'nn'
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None, 'linear'
                
        except Exception as e:
            logger.error(f"Failed to load batting model: {e}")
            return None, 'linear'

    def switch_sides(self):
        """攻守交代の処理"""
        # 攻守交代前に現在の打者の打順を1つ進める（イニング終了時の処理）
        # これにより、次のイニングは最後に打席に立った打者の次の打者から始まる
        current_batting_team = self.batting_team
        current_batting_team.next_batter()
        
        # 表から裏または裏から表への切り替え
        if self.is_top_inning:
            self._switch_to_bottom_inning()
        else:
            self._switch_to_next_inning()
        
        # ベースとアウト数をリセット
        self.bases = [None] * BASES_COUNT
        self.outs = 0
        
        # 攻守交代後の守備位置妥当性チェック
        if self.fielding_team is not None:
            is_ready, messages = self.fielding_team.check_defensive_readiness()
            
            if not is_ready:
                # エラーがある場合は操作を凍結
                self.defensive_position_error = True
                self.defensive_error_messages = messages
                print(f"\n⚠️ DEFENSIVE POSITION ERRORS for {self.fielding_team.name}:")
                for error in messages:
                    print(f"   ❌ {error}")
                print("   🚫 GAME ACTIONS FROZEN - Please fix defensive positions before continuing play.")
            elif messages:
                # 警告がある場合は情報として表示（ゲームは継続可能）
                self.defensive_position_error = False
                self.defensive_error_messages = []
                print(f"\n⚠️ Defensive Position Warnings for {self.fielding_team.name}:")
                for warning in messages:
                    print(f"   ⚠️ {warning}")
            else:
                # 問題がない場合は正常状態
                self.defensive_position_error = False
                self.defensive_error_messages = []
                #print(f"✅ {self.fielding_team.name} defensive positions are valid.")

    def is_game_action_allowed(self):
        """ゲームアクション（バッティングなど）が実行可能かどうかを判定"""
        if self.game_ended:
            return False, "Game has ended"
        
        if self.defensive_position_error:
            error_msg = f"Game actions are frozen due to defensive position errors: {', '.join(self.defensive_error_messages)}"
            return False, error_msg
            
        return True, ""

    def _switch_to_bottom_inning(self):
        """表から裏への切り替え"""
        # 9回表終了時にホームチームがリードしている場合、試合終了
        if self.inning >= INNINGS_PER_GAME and self.home_score > self.away_score:
            self.game_ended = True
            return
            
        self.is_top_inning = False
        self.batting_team = self.home_team
        self.fielding_team = self.away_team
        
        # バッター順番は継続（リセットしない）

    def _switch_to_next_inning(self):
        """裏から次のイニング表への切り替え"""
        # 9回以降の裏終了時：
        # - ホームチームがリードまたはビハインドの場合は試合終了
        # - 同点の場合のみ延長戦
        if self.inning >= INNINGS_PER_GAME and self.home_score != self.away_score:
            self.game_ended = True
            return
            
        # 延長戦制限チェック
        if self.inning >= MAX_EXTRA_INNINGS:
            self.game_ended = True
            logger.info(f"Game ended due to extra innings limit (inning {self.inning})")
            return
            
        self.is_top_inning = True
        self.inning += 1
        self.batting_team = self.away_team
        self.fielding_team = self.home_team
        
        # 新しいイニングのための得点枠を作成
        self.inning_scores[0].append(0)  # アウェイチーム
        self.inning_scores[1].append(0)  # ホームチーム
        
        self.outs = 0
        self.bases = [None, None, None]

    def add_out(self):
        """アウトカウントを増やし、必要に応じて攻守交代"""
        self.outs += 1
        if self.outs >= OUTS_PER_INNING:
            self.switch_sides()

    def can_bunt(self):
        """バントが可能かどうかを判定
        
        Returns:
            bool: バント可能な場合はTrue、不可能な場合はFalse
        """
        # バント不可能な条件をチェック
        if not any(runner is not None for runner in self.bases):
            return False  # ランナーがいない場合はバント不可
        
        if self.outs >= 2:
            return False  # 2アウトの場合はバント不可
        
        return True

    def execute_bunt(self, batter, pitcher):
        """バントを実行し、結果を返す
        
        Args:
            batter: 打者オブジェクト
            pitcher: 投手オブジェクト
            
        Returns:
            str: バントの結果を説明するメッセージ
        """
        from game_utils import BuntProcessor
        
        # BuntProcessorを使用してバント処理を実行
        bunt_processor = BuntProcessor(self)
        return bunt_processor.execute(batter, pitcher)

    def _add_runs(self, runs, batter):
        """得点の加算処理"""
        if self.batting_team == self.home_team:
            self.home_score += runs
            self.inning_scores[1][self.inning-1] += runs
            # サヨナラ勝ち判定
            if self.inning >= INNINGS_PER_GAME and not self.is_top_inning and self.home_score > self.away_score:
                self.game_ended = True
        else:
            self.away_score += runs
            self.inning_scores[0][self.inning-1] += runs
        
        self.fielding_team.current_pitcher.pitching_stats["ER"] += runs
        batter.stats["RBI"] += runs

    def calculate_result(self, batter, pitcher):
        """セイバーメトリクスに基づいた打席結果計算"""
        
        # 投手の現在の効果を計算（スタミナの影響を含む）
        pitcher_effectiveness = pitcher.get_effectiveness() / 100  # 0～1の範囲に標準化
        pitcher_effectiveness = 1.0  # 一旦、効果を無視
        
        # 打球確率の計算
        def calculate_prob(batter, pitcher, average):
            """打球確率の計算"""
            L = average / 100
            x = batter / 100
            y = pitcher / 100
            if x*y == 0:
                return 0.0
            # return 1 - (1 - x) * (1 - y) / (1 - L)
            return x*y*(1-L) / (x*y*(1-L) + (1-x)*(1-y)*L)

        k_prob = calculate_prob(batter.k_pct, pitcher.k_pct, 22.8)
        bb_prob = calculate_prob(batter.bb_pct, pitcher.bb_pct, 8.5)
        hard_prob = calculate_prob(batter.hard_pct, pitcher.hard_pct, 38.6)
        gb_prob = calculate_prob(batter.gb_pct, pitcher.gb_pct, 44.6)
        # k_prob = 22.8 / 100
        # bb_prob = 8.5 / 100
        # hard_prob = 38.6 / 100
        # gb_prob = 44.6 / 100
        other_prob = 1 - k_prob - bb_prob
        
        # 打席結果の確率計算
        features = ['K%', 'BB%', 'Hard%', 'GB%']
        status = [k_prob, bb_prob, hard_prob, gb_prob]
        data = dict(zip(features, status))
        
        if self.model is not None:
            prediction_result = predict_auto(self.model, data, self.model_type)
            for i in range(4):
                if prediction_result[i] < 0:
                    prediction_result[i] = 0
        else:
            # デフォルトの予測値（モデルが読み込めない場合）
            prediction_result = [0.15, 0.05, 0.01, 0.03, 0.76]  # 単打、二塁打、三塁打、本塁打、その他アウト
        
        single_prob = prediction_result[0]
        double_prob = prediction_result[1]
        triple_prob = prediction_result[2]
        hr_prob = prediction_result[3]
        out_woSO_prob = prediction_result[4]
        
        total = single_prob + double_prob + triple_prob + hr_prob + out_woSO_prob
        
        single_prob *= other_prob / total
        double_prob *= other_prob / total
        triple_prob *= other_prob / total
        hr_prob *= other_prob / total
        out_woSO_prob *= other_prob / total
        
        # 打者の左右と投手の左右の相性
        handedness_factor = 1.0  # 一旦、効果を無視
        
        # ゴロとフライの比率に基づいてアウトタイプを分割
        groundout_prob = out_woSO_prob * gb_prob
        flyout_prob = out_woSO_prob - groundout_prob
        
        # 最終的な確率分布
        probabilities = {
            "strikeout": k_prob,
            "walk": bb_prob,
            "single": single_prob,
            "double": double_prob,
            "triple": triple_prob,
            "home_run": hr_prob,
            "groundout": groundout_prob,
            "flyout": flyout_prob
        }
        
        # 確率の合計が1になるよう正規化
        total_prob = sum(probabilities.values())
        for key in probabilities:
            probabilities[key] /= total_prob
        
        # 結果の決定（確率に基づく乱数選択）
        roll = random.random()
        cumulative_prob = 0
        for result, prob in probabilities.items():
            cumulative_prob += prob
            if roll < cumulative_prob:
                return result
        
        # 念のため、デフォルト結果
        return "groundout"

    def apply_result(self, result, batter):
        """打席結果の適用と統計更新"""
        # 統計カウントの更新
        batter.stats["PA"] += 1
        
        if result == "strikeout":
            return self._handle_strikeout(batter)
        elif result == "walk":
            return self._handle_walk(batter)
        elif result == "single":
            return self._handle_single(batter)
        elif result == "double":
            return self._handle_double(batter)
        elif result == "triple":
            return self._handle_triple(batter)
        elif result == "home_run":
            return self._handle_home_run(batter)
        elif result == "groundout":
            return self._handle_groundout(batter)
        elif result == "flyout":
            return self._handle_flyout(batter)
        
        # デフォルト処理（通常は実行されない）
        self.add_out()
        return "Out."

    def _handle_strikeout(self, batter):
        """三振の処理"""
        self.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
        batter.stats["AB"] += 1
        batter.stats["SO"] += 1
        self.fielding_team.current_pitcher.pitching_stats["SO"] += 1
        self.add_out()
        return "Strike out"

    def _handle_walk(self, batter):
        """四球の処理"""
        batter.stats["BB"] += 1
        self.fielding_team.current_pitcher.pitching_stats["BB"] += 1

        # ランナー進塁はユーティリティに委譲
        from game_utils import RunnerEngine
        runs = RunnerEngine(self).apply_walk(batter)

        # 得点を加算
        if runs > 0:
            self._add_runs(runs, batter)
            return f"Walk - {runs} run(s) scored"
        
        return "Walk"

    def _handle_single(self, batter):
        """単打の処理"""
        batter.stats["AB"] += 1
        batter.stats["1B"] += 1
        self.fielding_team.current_pitcher.pitching_stats["H"] += 1

        # ランナー進塁はユーティリティに委譲
        from game_utils import RunnerEngine
        runs = RunnerEngine(self).apply_single(batter)

        if runs > 0:
            self._add_runs(runs, batter)
            return f"Single - {runs} run(s) scored"
        return "Single"

    def _handle_double(self, batter):
        """二塁打の処理"""
        batter.stats["AB"] += 1
        batter.stats["2B"] += 1
        self.fielding_team.current_pitcher.pitching_stats["H"] += 1

        # ランナー進塁はユーティリティに委譲
        from game_utils import RunnerEngine
        runs = RunnerEngine(self).apply_double(batter)

        if runs > 0:
            self._add_runs(runs, batter)
            return f"Double! {runs} run(s) scored!"
        return "Double!"

    def _handle_triple(self, batter):
        """三塁打の処理"""
        batter.stats["AB"] += 1
        batter.stats["3B"] += 1
        self.fielding_team.current_pitcher.pitching_stats["H"] += 1

        from game_utils import RunnerEngine
        runs = RunnerEngine(self).apply_triple(batter)
        if runs > 0:
            self._add_runs(runs, batter)
            return f"Triple! {runs} run(s) scored!"
        return "Triple!"

    def _handle_home_run(self, batter):
        """本塁打の処理"""
        batter.stats["AB"] += 1
        batter.stats["HR"] += 1
        
        self.fielding_team.current_pitcher.pitching_stats["H"] += 1
        self.fielding_team.current_pitcher.pitching_stats["HR"] += 1

        from game_utils import RunnerEngine
        runs = RunnerEngine(self).apply_home_run(batter)
        self._add_runs(runs, batter)
        if runs > 1:
            return f"{runs}-run home run!"
        else:
            return f"Solo home run!"

    def _handle_groundout(self, batter):
        """ゴロアウトの処理"""
        self.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
        batter.stats["AB"] += 1

        from game_utils import RunnerEngine
        runs, message = RunnerEngine(self).apply_groundout(batter)
        if runs > 0:
            self._add_runs(runs, batter)
        return message
    

    def _handle_flyout(self, batter):
        """フライアウトの処理"""
        self.fielding_team.current_pitcher.pitching_stats["IP"] += 1/3
        batter.stats["AB"] += 1

        from game_utils import RunnerEngine
        runs = RunnerEngine(self).apply_flyout(batter)
        self.add_out()  # 打者アウト
        if runs > 0:
            self._add_runs(runs, batter)
            return f"Sacrifice fly! {runs} run scored!"
        return "Flyout."
