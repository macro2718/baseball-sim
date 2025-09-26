"""Lineup management utilities for batting order and defensive positions."""

from baseball_sim.config import Positions

class LineupManager:
    def __init__(self, team):
        self.team = team
        self.defensive_positions = Positions.DEFENSIVE_POSITIONS
        self.all_positions = Positions.ALL_POSITIONS
        self.valid_positions = Positions.ALL_POSITIONS
        
    def find_player_by_position(self, position):
        """指定されたポジションの選手を見つける（後方互換性）"""
        for player in self.team.lineup:
            current_pos = player.current_position if hasattr(player, 'current_position') else player.position
            if current_pos == position:
                return player
        return None
        
    def validate_lineup(self):
        """ラインナップの妥当性をチェック"""
        if hasattr(self.team, 'validate_lineup'):
            return self.team.validate_lineup()
        
        # 後方互換性のための基本チェック
        errors = []
        
        if len(self.team.lineup) != 9:
            errors.append(f"Lineup has {len(self.team.lineup)} players, need 9")
        
        # ポジション重複チェック
        used_positions = []
        for player in self.team.lineup:
            pos = player.current_position if hasattr(player, 'current_position') else player.position
            if pos in used_positions:
                errors.append(f"Duplicate position: {pos}")
            used_positions.append(pos)
        
        # 適性チェック
        for player in self.team.lineup:
            pos = player.current_position if hasattr(player, 'current_position') else player.position
            if hasattr(player, 'can_play_position') and not player.can_play_position(pos):
                errors.append(f"{player.name} cannot play {pos}")
        
        return errors
    
    def set_player_position(self, player_index, new_position):
        """指定された選手のポジションを変更"""
        if not (0 <= player_index < len(self.team.lineup)):
            return False, "Invalid player index"
            
        if new_position not in self.valid_positions:
            return False, f"Invalid position: {new_position}"
            
        player = self.team.lineup[player_index]
        
        # 選手がそのポジションを守れるかチェック
        if not player.can_play_position(new_position):
            return False, f"{player.name} cannot play position {new_position}"
        
        # チームオブジェクトに守備位置管理システムがある場合はそれを使用
        if hasattr(self.team, 'set_player_position'):
            return self.team.set_player_position(player_index, new_position)
        else:
            # 後方互換性のため古いシステムを使用
            old_position = player.position
            
            # 新しいポジションに既に他の選手がいるかチェック
            existing_player = self.find_player_by_position(new_position)
            if existing_player and existing_player != player:
                # 相手の選手が元のポジションを守れるかチェック
                if not existing_player.can_play_position(old_position):
                    return False, f"{existing_player.name} cannot play position {old_position}"
                # ポジションを交換
                existing_player.position = old_position
                
            player.position = new_position
            return True, f"{player.name} moved from {old_position} to {new_position}"
        
    # Removed unused batting order/position swap utilities that are no longer called.
