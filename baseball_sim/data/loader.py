"""Load teams and player data from JSON sources."""

import json
from typing import Dict, List, Tuple

from baseball_sim.config import path_manager
from baseball_sim.data.player import Player
from baseball_sim.data.player_factory import PlayerFactory
from baseball_sim.gameplay.team import Team
from baseball_sim.infrastructure.logging_utils import log_error


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
        rotation_raw = team_data.get("rotation")
        rotation_names = []
        if isinstance(rotation_raw, list):
            for entry in rotation_raw:
                if isinstance(entry, dict):
                    name_value = entry.get("name")
                else:
                    name_value = entry
                name = str(name_value or "").strip()
                if name:
                    rotation_names.append(name)

        for pitcher_name in team_data["pitchers"]:
            team.add_pitcher(players_dict[pitcher_name])

        rotation_players = []
        if rotation_names:
            name_map = {pitcher.name: pitcher for pitcher in team.pitchers}
            for name in rotation_names:
                pitcher = name_map.get(name)
                if pitcher and pitcher not in rotation_players:
                    rotation_players.append(pitcher)

        team.set_pitcher_rotation(rotation_players)

    @staticmethod
    def setup_team_bench(team: Team, team_data: Dict, players_dict: Dict[str, Player]) -> None:
        """チームのベンチをセットアップ"""
        for player_name in team_data["bench"]:
            team.add_player_to_bench(players_dict[player_name])

    @classmethod
    def create_teams_from_data(
        cls,
        *,
        home_team_override: Dict | None = None,
        away_team_override: Dict | None = None,
    ) -> Tuple[Team, Team]:
        """JSONデータからチームを作成"""
        # パス管理
        player_data_path = path_manager.get_players_data_path()
        team_data_path = path_manager.get_teams_data_path()

        # JSONからデータを読み込む
        player_data = cls.load_json_data(player_data_path)
        team_data = cls.load_json_data(team_data_path)
        if not isinstance(team_data, dict):
            raise ValueError("Team data file is invalid or missing required structure.")

        home_team_data = home_team_override or team_data.get("home_team")
        away_team_data = away_team_override or team_data.get("away_team")

        if not isinstance(home_team_data, dict):
            raise ValueError("Home team data could not be loaded.")
        if not isinstance(away_team_data, dict):
            raise ValueError("Away team data could not be loaded.")

        # チームの作成
        home_team = Team(home_team_data["name"])
        away_team = Team(away_team_data["name"])

        # 選手辞書の作成（home/awayで別インスタンスにする）
        # 同じ選手名を共有すると、試合中のスタミナや成績が相互に影響してしまうため
        players_home = cls.create_players_dict(player_data)
        players_away = cls.create_players_dict(player_data)

        # 投手陣の設定
        cls.setup_team_pitchers(home_team, home_team_data, players_home)
        cls.setup_team_pitchers(away_team, away_team_data, players_away)

        # ラインナップの設定
        home_errors = cls.setup_team_lineup(home_team, home_team_data, players_home)
        away_errors = cls.setup_team_lineup(away_team, away_team_data, players_away)

        # ベンチの設定
        cls.setup_team_bench(home_team, home_team_data, players_home)
        cls.setup_team_bench(away_team, away_team_data, players_away)
        
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
