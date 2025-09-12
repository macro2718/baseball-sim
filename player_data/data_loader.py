"""
データローダーモジュール
JSONファイルからの選手・チームデータの読み込みと処理を担当
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple
from .player import Player, Pitcher
from main_code.core.team import Team
from main_code.config import path_manager
from .player_factory import PlayerFactory

try:
    from main_code.infra.error_handling import log_error
except ImportError:
    # フォールバック用デコレータ
    def log_error(func):
        return func


class DataLoader:
    """JSONファイルからチームと選手データを読み込むクラス"""
    
    @staticmethod
    @log_error
    def load_json_data(filepath: str) -> Dict:
        """JSONファイルからデータを読み込む"""
        if not path_manager.file_exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data file: {e}")

    @staticmethod
    def create_players_dict(player_data: Dict) -> Dict[str, Player]:
        """選手データから選手辞書を作成（PlayerFactoryに委譲）"""
        return PlayerFactory.create_players_dict(player_data)

    @staticmethod
    def setup_team_lineup(team: Team, team_data: Dict, players_dict: Dict[str, Player]) -> List[str]:
        """チームのラインナップをセットアップし、エラーメッセージを返す"""
        errors = []
        
        print(f"\n=== Setting up {team.name} lineup ===")
        for player_pos in team_data["lineup"]:
            player = players_dict[player_pos["name"]]
            position = player_pos["position"]
            
            # 適性チェック
            if not player.can_play_position(position):
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                print(f"Warning: {player.name} cannot play {position}, assigning to primary position {primary_pos}")
                position = primary_pos
            
            if not team.add_player_to_lineup(player, position):
                error_msg = f"Error: Could not add {player.name} to {position}"
                print(error_msg)
                errors.append(error_msg)
        
        return errors

    @staticmethod
    def setup_team_pitchers(team: Team, team_data: Dict, players_dict: Dict[str, Player]) -> None:
        """チームの投手陣をセットアップ"""
        for pitcher_name in team_data["pitchers"]:
            team.add_pitcher(players_dict[pitcher_name])

    @staticmethod
    def setup_team_bench(team: Team, team_data: Dict, players_dict: Dict[str, Player]) -> None:
        """チームのベンチをセットアップ"""
        for player_name in team_data["bench"]:
            team.add_player_to_bench(players_dict[player_name])

    @classmethod
    def create_teams_from_data(cls) -> Tuple[Team, Team]:
        """JSONデータからチームを作成"""
        # パス管理
        player_data_path = path_manager.get_players_data_path()
        team_data_path = path_manager.get_teams_data_path()
        
        # JSONからデータを読み込む
        player_data = cls.load_json_data(player_data_path)
        team_data = cls.load_json_data(team_data_path)
        
        # チームの作成
        home_team = Team(team_data["home_team"]["name"])
        away_team = Team(team_data["away_team"]["name"])
        
        # 選手辞書の作成
        players_dict = cls.create_players_dict(player_data)
        
        # 投手陣の設定
        cls.setup_team_pitchers(home_team, team_data["home_team"], players_dict)
        cls.setup_team_pitchers(away_team, team_data["away_team"], players_dict)
        
        # ラインナップの設定
        home_errors = cls.setup_team_lineup(home_team, team_data["home_team"], players_dict)
        away_errors = cls.setup_team_lineup(away_team, team_data["away_team"], players_dict)
        
        # ベンチの設定
        cls.setup_team_bench(home_team, team_data["home_team"], players_dict)
        cls.setup_team_bench(away_team, team_data["away_team"], players_dict)
        
        # ラインナップの妥当性チェック
        print(f"\n=== Validating lineups ===")
        home_validation_errors = home_team.validate_lineup()
        away_validation_errors = away_team.validate_lineup()
        
        if home_validation_errors:
            print(f"{home_team.name} lineup errors:")
            for error in home_validation_errors:
                print(f"  - {error}")
        else:
            print(f"{home_team.name} lineup is valid")
        
        if away_validation_errors:
            print(f"{away_team.name} lineup errors:")
            for error in away_validation_errors:
                print(f"  - {error}")
        else:
            print(f"{away_team.name} lineup is valid")
        
        return home_team, away_team
