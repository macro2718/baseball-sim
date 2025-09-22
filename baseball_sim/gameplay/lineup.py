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
        
    def swap_positions(self, player1_index, player2_index):
        """2人の選手のポジションを交換"""
        if not (0 <= player1_index < len(self.team.lineup) and 
                0 <= player2_index < len(self.team.lineup)):
            return False, "Invalid player indices"
            
        if player1_index == player2_index:
            return False, "Cannot swap player with themselves"
            
        player1 = self.team.lineup[player1_index]
        player2 = self.team.lineup[player2_index]
        
        # 選手がお互いのポジションを守れるかチェック
        if not player1.can_play_position(player2.current_position):
            return False, f"{player1.name} cannot play position {player2.current_position}"
        if not player2.can_play_position(player1.current_position):
            return False, f"{player2.name} cannot play position {player1.current_position}"
        
        # DHとの交換は制限
        if player1.current_position == "DH" or player2.current_position == "DH":
            return False, "Cannot swap DH position with field positions"
        
        # チームオブジェクトに守備位置管理システムがある場合はそれを使用
        if hasattr(self.team, 'switch_positions'):
            return self.team.switch_positions(player1_index, player2_index)
        else:
            # 後方互換性のため古いシステムを使用
            # ポジションを交換
            player1.position, player2.position = player2.position, player1.position
            
            return True, f"Swapped positions: {player1.name} <-> {player2.name}"
        
    def move_player_in_batting_order(self, from_index, to_index):
        """打順を変更"""
        if not (0 <= from_index < len(self.team.lineup) and
                0 <= to_index < len(self.team.lineup)):
            return False, "Invalid batting order indices"

        if from_index == to_index:
            return False, "Player is already in that position"

        player = self.team.lineup.pop(from_index)
        self.team.lineup.insert(to_index, player)

        return True, f"Moved {player.name} from {from_index+1} to {to_index+1} in batting order"

    def swap_players_in_batting_order(self, index1, index2):
        """2人の選手の打順を入れ替える"""
        if not (0 <= index1 < len(self.team.lineup) and 0 <= index2 < len(self.team.lineup)):
            return False, "Invalid batting order indices"

        if index1 == index2:
            return False, "Cannot swap the same player"

        player1 = self.team.lineup[index1]
        player2 = self.team.lineup[index2]
        self.team.lineup[index1], self.team.lineup[index2] = player2, player1

        return True, f"Swapped {player1.name} and {player2.name}"
