"""Domain objects representing players and pitchers."""

import random

from baseball_sim.config import Positions, StatColumns
from baseball_sim.gameplay.statistics import StatsCalculator

class Player:
    def __init__(self, name, eligible_positions, k_pct, bb_pct, hard_pct, speed, gb_pct, fielding_skill, 
                 bats="R"):
        self.name = name
        self.eligible_positions = eligible_positions  # 守備可能ポジションのリスト
        self.current_position = None                  # 現在の守備位置（試合中に設定される）
        
        # 新しいパラメータセット（セイバーメトリクス向け）
        self.bats = bats          # 打席位置（"R"または"L"）
        self.k_pct = k_pct        # 三振率（0-100）
        self.bb_pct = bb_pct      # 四球率（0-100）
        self.hard_pct = hard_pct  # 強い打球を打つ確率（0-100）
        self.speed = speed        # 走力（塁間移動時間、秒単位で3.8-4.8程度）
        self.gb_pct = gb_pct  # ゴロ率
        self.fielding_skill = fielding_skill  # 守備力（0-100）
        
        self.stats = {
            StatColumns.PLATE_APPEARANCES: 0,
            StatColumns.AT_BATS: 0,
            StatColumns.SINGLES: 0,
            StatColumns.DOUBLES: 0,
            StatColumns.TRIPLES: 0,
            StatColumns.HOME_RUNS: 0,
            StatColumns.WALKS: 0,
            StatColumns.STRIKEOUTS: 0,
            StatColumns.RUNS_BATTED_IN: 0
        }

    def __str__(self):
        """選手の文字列表現を返す。全ての守備可能ポジションと現在の守備位置を含む。"""
        eligible_pos_str = ", ".join(self.eligible_positions) if self.eligible_positions else "N/A"
        
        pos_info_parts = f"Eligible: {eligible_pos_str}"
            
        return f"{self.current_position} | {self.name} [{pos_info_parts}]"
    
    def can_play_position(self, position):
        """指定されたポジションを守れるかチェック"""
        return position in self.eligible_positions
    
    def get_display_eligible_positions(self):
        """表示用の守備可能位置を返す（DHを除外）"""
        return [pos for pos in self.eligible_positions if pos != "DH"]
    
    @property
    def position(self):
        """後方互換性のためのプロパティ"""
        return self.current_position if self.current_position else (self.eligible_positions[0] if self.eligible_positions else "N/A")
    
    @position.setter
    def position(self, value):
        """後方互換性のためのセッター"""
        self.current_position = value
    
    def get_avg(self):
        """打率を計算（統一メソッドを使用）"""
        total_hits = self.stats["1B"] + self.stats.get("2B", 0) + self.stats.get("3B", 0) + self.stats.get("HR", 0)
        return StatsCalculator.calculate_batting_average(total_hits, self.stats["AB"])
    
    def get_obp(self):
        """出塁率を計算（統一メソッドを使用）"""
        total_hits = self.stats["1B"] + self.stats.get("2B", 0) + self.stats.get("3B", 0) + self.stats.get("HR", 0)
        return StatsCalculator.calculate_obp(total_hits, self.stats["BB"], self.stats["AB"])
    
    def get_slg(self):
        """長打率を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_slg(
            self.stats["1B"], 
            self.stats.get("2B", 0), 
            self.stats.get("3B", 0), 
            self.stats.get("HR", 0), 
            self.stats["AB"]
        )
    
    def get_ops(self):
        """OPS を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_ops(self.get_obp(), self.get_slg())
    
    def get_babip(self):
        """BABIP（本塁打を除く打球の安打率）を計算（統一メソッドを使用）"""
        total_hits = self.stats["1B"] + self.stats.get("2B", 0) + self.stats.get("3B", 0) + self.stats.get("HR", 0)
        return StatsCalculator.calculate_babip(total_hits, self.stats.get("HR", 0), self.stats["AB"], self.stats["SO"])


class Pitcher(Player):
    def __init__(self, name, k_pct, bb_pct, hard_pct, gb_pct, stamina, throws="R", pitcher_type="SP"):
        # Player(name, eligible_positions, k_pct, bb_pct, hard_pct, speed, gb_pct, fielding_skill, bats)
        super().__init__(name, ["P"], k_pct, bb_pct, hard_pct, 4.3, gb_pct, 70, throws)
        self.k_pct = k_pct      # 三振率（0-100）
        self.bb_pct = bb_pct    # 四球率（0-100）
        self.hard_pct = hard_pct  # 強い打球を打たれる確率（0-100）
        self.gb_pct = gb_pct    # ゴロ打率（0-100）
        self.stamina = stamina  # スタミナ（0-100）
        self.current_stamina = stamina
        self.throws = throws    # 投球腕（"R"または"L"）
        self.pitcher_type = pitcher_type  # "SP"（先発投手）または"RP"（中継ぎ投手）
        
        self.pitching_stats = {"IP": 0, "H": 0, "R": 0, "ER": 0, "BB": 0, "SO": 0, "HR": 0}

    def __str__(self):
        return f"{self.name} ({self.pitcher_type}) - Stamina: {int(self.current_stamina)}%"

    def get_display_eligible_positions(self):
        """表示用には投手タイプ（SP/RP）を返す。"""
        return [self.pitcher_type]
    
    def get_effectiveness(self):
        """現在の投球効果を返す（スタミナ影響下）"""
        return self.current_stamina / self.stamina * 100
    
    def decrease_stamina(self):
        """各打者後にスタミナを減少させる"""
        self.current_stamina = max(0, self.current_stamina - random.uniform(3, 4.5))
    
    def get_era(self):
        """防御率を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_era(self.pitching_stats["ER"], self.pitching_stats["IP"])
    
    def get_whip(self):
        """WHIP を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_whip(self.pitching_stats["H"], self.pitching_stats["BB"], self.pitching_stats["IP"])
    
    def get_k_per_9(self):
        """K/9 を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_k_per_9(self.pitching_stats["SO"], self.pitching_stats["IP"])
    
    def get_bb_per_9(self):
        """BB/9 を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_bb_per_9(self.pitching_stats["BB"], self.pitching_stats["IP"])
    
    def get_hr_per_9(self):
        """HR/9 を計算（統一メソッドを使用）"""
        return StatsCalculator.calculate_hr_per_9(self.pitching_stats["HR"], self.pitching_stats["IP"])
