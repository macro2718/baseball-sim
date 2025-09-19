"""Factories for creating player instances used throughout the app."""

import random
from typing import Dict, List

from baseball_sim.data.player import Pitcher, Player


class PlayerFactory:
    """プレイヤー作成の統一クラス"""
    
    # デフォルトパラメータ（generate_team_data.pyから移動）
    PITCHER_PARAMS = {
        "k_pct": 22.8,             # K%（三振率）
        "bb_pct": 8.5,             # BB%（四球率）
        "hard_pct": 38.6,          # Hard%（強い打球を打たれる割合）
        "gb_pct": 44.6,            # GB%（ゴロ打率）
        "stamina": 80              # スタミナ
    }
    
    BATTER_PARAMS = {
        "k_pct": 22.8,             # K%（三振率）
        "bb_pct": 8.5,             # BB%（四球率）
        "hard_pct": 38.6,          # Hard%（強い打球の割合）
        "gb_pct": 44.6,            # GB%（ゴロ打率）
        "speed": 4.3,              # 走力（塁間走速、秒）
        "fielding_skill": 100      # 守備力
    }
    
    @staticmethod
    def _get_random_parameter(base, variation=0.15, use_sample_mode=False):
        """パラメータに適度なランダム性を追加"""
        if use_sample_mode:
            # サンプルモードでは固定値を返す
            return base
        if random.random() < 0.05:  # 5%の確率で大きく変化
            return max(0, base * (1 + random.gauss(mu=0.0, sigma=variation * 2)))
        else:
            return max(0, base * (1 + random.gauss(mu=0.0, sigma=variation)))
    
    @classmethod
    def create_pitcher(cls, name: str, pitcher_type: str = "SP", use_sample_mode: bool = False, **kwargs) -> Pitcher:
        """統一された投手作成メソッド"""
        # 先発投手と中継ぎ投手で異なるパラメータを設定
        if pitcher_type == "SP":
            stamina_bonus = 10
            bb_modifier = 1.0
        else:  # RP (中継ぎ投手)
            stamina_bonus = -15
            bb_modifier = 0.9
        
        # パラメータを計算（kwargsで上書き可能）
        k_pct = kwargs.get('k_pct', cls._get_random_parameter(cls.PITCHER_PARAMS["k_pct"], use_sample_mode=use_sample_mode))
        bb_pct = kwargs.get('bb_pct', cls._get_random_parameter(cls.PITCHER_PARAMS["bb_pct"], use_sample_mode=use_sample_mode) * bb_modifier)
        hard_pct = kwargs.get('hard_pct', cls._get_random_parameter(cls.PITCHER_PARAMS["hard_pct"], use_sample_mode=use_sample_mode))
        gb_pct = kwargs.get('gb_pct', cls._get_random_parameter(cls.PITCHER_PARAMS["gb_pct"], use_sample_mode=use_sample_mode))
        stamina = kwargs.get('stamina', max(30, cls._get_random_parameter(cls.PITCHER_PARAMS["stamina"], use_sample_mode=use_sample_mode) + stamina_bonus))
        throws = kwargs.get('throws', random.choice(["R", "L"]))
        
        return Pitcher(name, k_pct, bb_pct, hard_pct, gb_pct, stamina, throws, pitcher_type)
    
    @classmethod
    def create_batter(cls, name: str, eligible_positions: List[str], use_sample_mode: bool = False, **kwargs) -> Player:
        """統一された野手作成メソッド"""
        # パラメータを計算（kwargsで上書き可能）
        k_pct = kwargs.get('k_pct', cls._get_random_parameter(cls.BATTER_PARAMS["k_pct"], use_sample_mode=use_sample_mode))
        bb_pct = kwargs.get('bb_pct', cls._get_random_parameter(cls.BATTER_PARAMS["bb_pct"], use_sample_mode=use_sample_mode))
        hard_pct = kwargs.get('hard_pct', cls._get_random_parameter(cls.BATTER_PARAMS["hard_pct"], use_sample_mode=use_sample_mode))
        gb_pct = kwargs.get('gb_pct', cls._get_random_parameter(cls.BATTER_PARAMS["gb_pct"], use_sample_mode=use_sample_mode))
        speed = kwargs.get('speed', cls._get_random_parameter(cls.BATTER_PARAMS["speed"], 0.05, use_sample_mode=use_sample_mode))
        fielding_skill = kwargs.get('fielding_skill', cls._get_random_parameter(cls.BATTER_PARAMS["fielding_skill"], use_sample_mode=use_sample_mode))
        bats = kwargs.get('bats', random.choice(["R", "L"]))
        
        return Player(name, eligible_positions, k_pct, bb_pct, hard_pct, speed, gb_pct, fielding_skill, bats)
    
    @classmethod
    def create_pitcher_from_data(cls, pitcher_data: Dict) -> Pitcher:
        """データ辞書から投手を作成（data_loader.py用）"""
        return Pitcher(
            pitcher_data["name"], 
            pitcher_data["k_pct"], 
            pitcher_data["bb_pct"], 
            pitcher_data["hard_pct"], 
            pitcher_data["gb_pct"], 
            pitcher_data["stamina"], 
            pitcher_data["throws"],
            pitcher_data.get("pitcher_type", "SP")
        )
    
    @classmethod
    def create_batter_from_data(cls, batter_data: Dict) -> Player:
        """データ辞書から野手を作成（data_loader.py用）"""
        return Player(
            batter_data["name"],
            batter_data["eligible_positions"],
            batter_data["k_pct"],
            batter_data["bb_pct"],
            batter_data["hard_pct"],
            batter_data["speed"],
            batter_data["gb_pct"],
            batter_data["fielding_skill"],
            batter_data["bats"]
        )
    
    @classmethod
    def create_players_dict(cls, player_data: Dict) -> Dict[str, Player]:
        """選手データから選手辞書を作成（統一メソッド）"""
        players_dict = {}
        
        # 投手を作成して辞書に登録
        for p_data in player_data["pitchers"]:
            pitcher = cls.create_pitcher_from_data(p_data)
            players_dict[p_data["name"]] = pitcher
        
        # 野手を作成して辞書に登録
        for b_data in player_data["batters"]:
            batter = cls.create_batter_from_data(b_data)
            players_dict[b_data["name"]] = batter
        
        return players_dict
    
    @classmethod
    def create_team_roster(cls, pitcher_names: List[str], batter_names: List[str], 
                          use_sample_mode: bool = False) -> List[Player]:
        """チーム全体のロスター作成"""
        roster = []
        
        # 投手を作成
        starters_count = len(pitcher_names) // 3  # 約1/3が先発投手
        for i, name in enumerate(pitcher_names):
            pitcher_type = "SP" if i < starters_count else "RP"
            pitcher = cls.create_pitcher(name, pitcher_type, use_sample_mode)
            roster.append(pitcher)
        
        # 野手を作成 - 基本的なポジション設定
        POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        
        def get_eligible_positions(primary_position):
            """主ポジションに基づいて守備可能ポジションを決定"""
            eligible = [primary_position]
            # DHは全員が適性を持つ
            eligible.append("DH")
            
            # ポジション間の適性ルール
            if primary_position in ["C"]:
                eligible.extend(["1B"])
            elif primary_position in ["1B", "3B"]:
                eligible.extend(["LF", "RF"])
            elif primary_position in ["2B", "SS"]:
                eligible.extend(["CF"])
            elif primary_position in ["LF", "CF", "RF"]:
                eligible.extend(["LF", "CF", "RF"])
            
            return list(set(eligible))
        
        for i, name in enumerate(batter_names):
            if i < len(POSITIONS):
                # 最初の8人は各ポジション1人ずつ
                primary_position = POSITIONS[i]
                eligible_positions = get_eligible_positions(primary_position)
            else:
                # 残りの選手はランダムポジション
                primary_position = random.choice(POSITIONS)
                eligible_positions = get_eligible_positions(primary_position)
            
            batter = cls.create_batter(name, eligible_positions, use_sample_mode)
            roster.append(batter)
        
        return roster
