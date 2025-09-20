"""Team creation and management helpers used by the GUI."""

from baseball_sim.config import setup_project_environment
from baseball_sim.data.loader import DataLoader
from baseball_sim.gameplay.lineup import LineupManager
from baseball_sim.gameplay.substitutions import SubstitutionManager

setup_project_environment()


class TeamManager:
    """
    GUI用のチーム管理クラス
    チームの作成、ラインナップ管理、バリデーションを担当
    """
    
    def __init__(self):
        self.home_team = None
        self.away_team = None
        self._lineup_managers = {}
        self._substitution_managers = {}
        
    def create_teams(self):
        """チームを作成"""
        try:
            self.home_team, self.away_team = DataLoader.create_teams_from_data()
            
            # LineupManagerとSubstitutionManagerの初期化
            self._lineup_managers['home'] = LineupManager(self.home_team)
            self._lineup_managers['away'] = LineupManager(self.away_team)
            self._substitution_managers['home'] = SubstitutionManager(self.home_team)
            self._substitution_managers['away'] = SubstitutionManager(self.away_team)
            
            return True, f"Teams created: {self.home_team.name} vs {self.away_team.name}"
            
        except Exception as e:
            return False, f"Failed to create teams: {e}"
    
    def get_teams(self):
        """現在のチームを取得"""
        return self.home_team, self.away_team
    
    def has_teams(self):
        """チームが作成済みかチェック"""
        return self.home_team is not None and self.away_team is not None
    
    def get_lineup_manager(self, team_type):
        """指定されたチームのLineupManagerを取得
        
        Args:
            team_type (str): 'home' または 'away'
        """
        return self._lineup_managers.get(team_type)
    
    def validate_lineup(self, team_type):
        """指定されたチームのラインナップをバリデート
        
        Args:
            team_type (str): 'home' または 'away'
            
        Returns:
            list: エラーリスト（空の場合は問題なし）
        """
        if team_type == 'home' and self.home_team:
            return self.home_team.validate_lineup()
        elif team_type == 'away' and self.away_team:
            return self.away_team.validate_lineup()
        else:
            return ["Team not found"]
    
    def validate_all_lineups(self):
        """すべてのチームのラインナップをバリデート
        
        Returns:
            dict: {'home': errors, 'away': errors}
        """
        return {
            'home': self.validate_lineup('home'),
            'away': self.validate_lineup('away')
        }
    
    def get_team_by_type(self, team_type):
        """タイプでチームを取得
        
        Args:
            team_type (str): 'home' または 'away'
        """
        if team_type == 'home':
            return self.home_team
        elif team_type == 'away':
            return self.away_team
        else:
            return None
    
    def reset_lineups(self):
        """すべてのラインナップをリセット"""
        for manager in self._lineup_managers.values():
            if hasattr(manager, '_reset_positions'):
                manager._reset_positions()

    def reset_teams(self):
        """チーム情報をクリアして再読み込み可能にする"""
        self.home_team = None
        self.away_team = None
        self._lineup_managers.clear()
        self._substitution_managers.clear()
    
    def get_team_status(self, team_type):
        """チームの状態を取得
        
        Args:
            team_type (str): 'home' または 'away'
            
        Returns:
            dict: status情報
        """
        team = self.get_team_by_type(team_type)
        if not team:
            return {'valid': False, 'message': 'Team not found', 'errors': []}
        
        errors = self.validate_lineup(team_type)
        valid = len(errors) == 0
        
        if valid:
            message = "✓ Ready to play"
            color = "green"
        else:
            message = f"⚠ {len(errors)} issue(s)"
            color = "orange"
            
        return {
            'valid': valid,
            'message': message,
            'color': color,
            'errors': errors,
            'team_name': team.name
        }
    
    def setup_team_lineup(self, team_type):
        """指定されたチームのラインナップ設定を開始
        
        Args:
            team_type (str): 'home' または 'away'
        """
        manager = self.get_lineup_manager(team_type)
        if manager:
            # LineupManagerのGUI対応メソッドを呼び出し
            # （実装はgui_strategy.pyのロジックを使用）
            return manager
        return None
    
    def get_substitution_manager(self, team_type):
        """指定されたチームのSubstitutionManagerを取得
        
        Args:
            team_type (str): 'home' または 'away'
        """
        return self._substitution_managers.get(team_type)
    
    def get_substitution_info(self, team_type):
        """選手交代情報を取得
        
        Args:
            team_type (str): 'home' または 'away'
            
        Returns:
            dict: 交代情報
        """
        substitution_manager = self.get_substitution_manager(team_type)
        if substitution_manager:
            return substitution_manager.get_substitution_info()
        return {}
