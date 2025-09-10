"""
Substitution Manager
選手交代・ポジション変更の統一管理モジュール
GUIとターミナルモードの両方で使用される選手交代機能を提供
"""

from typing import Optional, Tuple, List, Dict, Any
from lineup_manager import LineupManager
import itertools


class SubstitutionManager:
    """
    選手交代を統一管理するクラス
    GUIとターミナルの両方のモードから使用される
    """
    
    def __init__(self, team):
        """
        Args:
            team: チームオブジェクト
        """
        self.team = team
        self.lineup_manager = LineupManager(team)
    
    def get_available_bench_players(self) -> List:
        """出場可能なベンチ選手のリストを取得"""
        return self.team.get_available_bench_players()
    
    def get_available_pitchers(self) -> List:
        """出場可能な投手のリストを取得"""
        return self.team.get_available_pitchers()
    
    def validate_substitution(self, substitute_player, position: Optional[str] = None) -> Tuple[bool, str]:
        """
        選手交代の妥当性をチェック
        
        Args:
            substitute_player: 代替選手
            position: ポジション（指定されない場合は現在のポジション）
        
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        if substitute_player not in self.team.bench:
            return False, f"{substitute_player.name} is not available on the bench"
        
        if substitute_player in self.team.ejected_players:
            return False, f"{substitute_player.name} has already been ejected and cannot re-enter"
        
        if position and not substitute_player.can_play_position(position):
            return False, f"{substitute_player.name} cannot play position {position}"
        
        return True, "Substitution is valid"
    
    def execute_pinch_hit(self, bench_player_index: int, lineup_index: int) -> Tuple[bool, str]:
        """
        代打を実行
        
        Args:
            bench_player_index: ベンチ選手のインデックス
            lineup_index: ラインナップでの選手のインデックス
        
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        try:
            available_bench = self.get_available_bench_players()
            if bench_player_index >= len(available_bench):
                return False, "Invalid bench player index"
            
            substitute_player = available_bench[bench_player_index]
            
            # 妥当性チェック
            valid, message = self.validate_substitution(substitute_player)
            if not valid:
                return False, message
            
            # 守備位置の整合性チェック（代打実行前に行う）
            defensive_valid, defensive_message = self.validate_pinch_hit_defense(substitute_player, lineup_index)
            if not defensive_valid:
                return False, defensive_message
            
            # 代打実行
            success, result_message = self.team.pinch_hit(lineup_index, substitute_player)
            
            return success, result_message
            
        except Exception as e:
            return False, f"Pinch hit error: {str(e)}"
    
    def execute_pitcher_change(self, pitcher_index: int) -> Tuple[bool, str]:
        """
        投手交代を実行
        
        Args:
            pitcher_index: 投手のインデックス
        
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        try:
            available_pitchers = self.get_available_pitchers()
            if pitcher_index >= len(available_pitchers):
                return False, "Invalid pitcher index"
            
            new_pitcher = available_pitchers[pitcher_index]
            
            # 妥当性チェック
            valid, message = self.validate_substitution(new_pitcher)
            if not valid:
                return False, message
            
            # 投手交代実行
            success, result_message = self.team.change_pitcher(new_pitcher)
            
            return success, result_message
            
        except Exception as e:
            return False, f"Pitcher change error: {str(e)}"
    
    def execute_defensive_substitution(self, bench_player_index: int, lineup_index: int) -> Tuple[bool, str]:
        """
        守備交代を実行
        
        Args:
            bench_player_index: ベンチ選手のインデックス
            lineup_index: ラインナップでの選手のインデックス
        
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        try:
            available_bench = self.get_available_bench_players()
            if bench_player_index >= len(available_bench):
                return False, "Invalid bench player index"
            
            substitute_player = available_bench[bench_player_index]
            
            if lineup_index >= len(self.team.lineup):
                return False, "Invalid lineup index"
            
            # 現在の選手のポジションを取得
            original_player = self.team.lineup[lineup_index]
            position = original_player.current_position
            
            # 妥当性チェック
            valid, message = self.validate_substitution(substitute_player, position)
            if not valid:
                return False, message
            
            # 守備交代実行
            success, result_message = self.team.substitute_player(lineup_index, substitute_player)
            
            return success, result_message
            
        except Exception as e:
            return False, f"Defensive substitution error: {str(e)}"
    
    def execute_position_switch(self, player1_index: int, player2_index: int) -> Tuple[bool, str]:
        """
        2人の選手のポジションを交換
        
        Args:
            player1_index: 1人目の選手のインデックス
            player2_index: 2人目の選手のインデックス
        
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        try:
            if not (0 <= player1_index < len(self.team.lineup) and 
                    0 <= player2_index < len(self.team.lineup)):
                return False, "Invalid player indices"
            
            player1 = self.team.lineup[player1_index]
            player2 = self.team.lineup[player2_index]
            
            # DHとの交換は制限
            if player1.current_position == "DH" or player2.current_position == "DH":
                return False, "Cannot switch positions with DH"
            
            # 選手がお互いのポジションを守れるかチェック
            if not player1.can_play_position(player2.current_position):
                return False, f"{player1.name} cannot play position {player2.current_position}"
            if not player2.can_play_position(player1.current_position):
                return False, f"{player2.name} cannot play position {player1.current_position}"
            
            # ポジション交換実行
            success = self.team.switch_positions(player1_index, player2_index)
            
            if success:
                return True, f"Position switch completed: {player1.name} <-> {player2.name}"
            else:
                return False, "Position switch failed"
                
        except Exception as e:
            return False, f"Position switch error: {str(e)}"
    
    def execute_individual_position_change(self, player_index: int, new_position: str) -> Tuple[bool, str]:
        """
        個別選手のポジション変更
        
        Args:
            player_index: 選手のインデックス
            new_position: 新しいポジション
        
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        try:
            if not (0 <= player_index < len(self.team.lineup)):
                return False, "Invalid player index"
            
            player = self.team.lineup[player_index]
            current_position = player.current_position
            
            # 適性チェック
            if not player.can_play_position(new_position):
                return False, f"{player.name} cannot play {new_position}"
            
            # 同じポジションの他の選手がいるかチェック
            existing_player = None
            for i, p in enumerate(self.team.lineup):
                if i != player_index and p.current_position == new_position:
                    existing_player = p
                    break
            
            if existing_player:
                # 既存の選手が元のポジションを守れるかチェック
                if not existing_player.can_play_position(current_position):
                    return False, f"{existing_player.name} cannot play {current_position}"
                
                # ポジション交換
                existing_player.current_position = current_position
                
                # 守備位置辞書も更新
                if hasattr(self.team, 'defensive_positions'):
                    self.team.defensive_positions[current_position] = existing_player
            
            # ポジション変更実行
            player.current_position = new_position
            
            # 守備位置辞書も更新
            if hasattr(self.team, 'defensive_positions'):
                self.team.defensive_positions[new_position] = player
            
            if existing_player:
                return True, f"Position exchange: {player.name} ({current_position} → {new_position}), {existing_player.name} ({new_position} → {current_position})"
            else:
                return True, f"Position change: {player.name} ({current_position} → {new_position})"
            
        except Exception as e:
            return False, f"Position change error: {str(e)}"
    
    def get_substitution_info(self) -> Dict[str, Any]:
        """
        現在の交代可能状況の情報を取得
        
        Returns:
            Dict[str, Any]: 交代情報
        """
        return {
            'available_bench': self.get_available_bench_players(),
            'available_pitchers': self.get_available_pitchers(),
            'current_lineup': self.team.lineup.copy(),
            'current_pitcher': self.team.current_pitcher,
            'ejected_players': self.team.ejected_players.copy(),
            'validation_errors': self.team.validate_lineup() if hasattr(self.team, 'validate_lineup') else []
        }
    
    def get_position_info(self) -> Dict[str, Any]:
        """
        現在のポジション情報を取得
        
        Returns:
            Dict[str, Any]: ポジション情報
        """
        position_info = {}
        
        for i, player in enumerate(self.team.lineup):
            position_info[player.current_position] = {
                'player': player,
                'lineup_index': i,
                'eligible_positions': player.eligible_positions if hasattr(player, 'eligible_positions') else [],
                'can_switch': player.current_position != "DH"
            }
        
        return position_info
    
    def format_player_info(self, player, include_positions: bool = True) -> str:
        """
        選手情報をフォーマットして文字列で返す
        
        Args:
            player: 選手オブジェクト
            include_positions: ポジション情報を含めるかどうか
        
        Returns:
            str: フォーマット済み選手情報
        """
        info = player.name
        
        if include_positions:
            primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
            current_pos = player.current_position if hasattr(player, 'current_position') else primary_pos
            
            if hasattr(player, 'get_display_eligible_positions'):
                eligible_positions = ", ".join(player.get_display_eligible_positions())
            else:
                eligible_positions = ", ".join(player.eligible_positions) if player.eligible_positions else "N/A"
            
            info += f" ({current_pos}) [Eligible: {eligible_positions}]"
        
        return info
    
    def get_formatted_lineup(self) -> List[str]:
        """
        フォーマット済みのラインナップリストを取得
        
        Returns:
            List[str]: フォーマット済みラインナップ
        """
        formatted_lineup = []
        for i, player in enumerate(self.team.lineup):
            formatted_lineup.append(f"{i+1}. {self.format_player_info(player)}")
        
        return formatted_lineup
    
    def get_formatted_bench(self) -> List[str]:
        """
        フォーマット済みのベンチ選手リストを取得
        
        Returns:
            List[str]: フォーマット済みベンチリスト
        """
        formatted_bench = []
        available_bench = self.get_available_bench_players()
        
        for i, player in enumerate(available_bench):
            formatted_bench.append(f"{i+1}. {self.format_player_info(player)}")
        
        return formatted_bench
    
    def get_formatted_pitchers(self) -> List[str]:
        """
        フォーマット済みの投手リストを取得
        
        Returns:
            List[str]: フォーマット済み投手リスト
        """
        formatted_pitchers = []
        available_pitchers = self.get_available_pitchers()
        
        for i, pitcher in enumerate(available_pitchers):
            pitcher_type = pitcher.pitcher_type if hasattr(pitcher, 'pitcher_type') else 'P'
            formatted_pitchers.append(f"{i+1}. {pitcher.name} ({pitcher_type})")
        
        return formatted_pitchers
    
    def validate_pinch_hit_defense(self, substitute_player, lineup_index: int) -> Tuple[bool, str]:
        """
        代打時の守備位置整合性をチェック（改良版：最大二部マッチング）
        
        Args:
            substitute_player: 代打選手
            lineup_index: 代打される選手のインデックス
            
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        if lineup_index >= len(self.team.lineup):
            return False, "Invalid lineup index"
        
        original_player = self.team.lineup[lineup_index]
        original_position = original_player.current_position
        
        # DHへの代打は問題なし
        if original_position == "DH":
            return True, ""
        
        # 代打選手がそのポジションを守れる場合は問題なし
        if substitute_player.can_play_position(original_position):
            return True, ""
        
        # 守備配置の再構成が可能かチェック
        return self._check_defensive_assignment_feasibility(original_player, substitute_player)
    
    def _check_defensive_assignment_feasibility(self, original_player, substitute_player):
        """
        守備位置の再配置可能性をチェック（最大二部マッチング問題として解決）
        
        Args:
            original_player: 代打される選手
            substitute_player: 代打選手
            
        Returns:
            Tuple[bool, str]: (可能かどうか, エラーメッセージ)
        """
        # 必要な守備位置（投手以外の8ポジション）
        required_positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        
        # 利用可能な選手リスト（交代選手を除く現在の野手 + 全ての控え選手）
        available_players = []
        
        # 現在出場中の選手（代打対象を除く）
        for player in self.team.lineup:
            if player != original_player:
                available_players.append(player)
        
        # 利用可能な控え選手全員を追加
        # substitute_playerもこの中に含まれている
        available_players.extend(self.get_available_bench_players())
        
        # 8人の選手で8つのポジションを守れるかチェック
        if len(available_players) < len(required_positions):
            return False, f"Not enough players to cover all positions (need {len(required_positions)}, have {len(available_players)})"
        
        # 各選手が守れるポジションのマッピングを作成
        player_positions = {}
        for player in available_players:
            player_positions[player.name] = []
            for pos in required_positions:
                if player.can_play_position(pos):
                    player_positions[player.name].append(pos)
        
        # 最大二部マッチングアルゴリズムを使用して配置可能かチェック
        match_result = self._find_maximum_bipartite_matching(
            [player.name for player in available_players], 
            required_positions, 
            player_positions
        )
        
        # 全てのポジションが埋まるかチェック
        if len(match_result) == len(required_positions):
            return True, ""
        else:
            assigned_positions = set(match_result.values())
            unassigned_positions = set(required_positions) - assigned_positions
            return False, f"Cannot assign players to positions: {', '.join(unassigned_positions)}"
    
    def _find_maximum_bipartite_matching(self, players, positions, player_positions):
        """
        最大二部マッチングを見つける。選手がポジション数より多い場合は組み合わせを試す。
        標準的な増加路アルゴリズム（DFS実装）を使用。
        """
        
        def find_perfect_matching(candidate_players, target_positions, mapping):
            """与えられた選手とポジションで完全マッチングを試みる"""
            pos_to_player = {}  # position -> player

            def dfs(player, visited_positions):
                """増加路を探索する"""
                for pos in mapping.get(player, []):
                    if pos not in visited_positions:
                        visited_positions.add(pos)
                        if pos not in pos_to_player or dfs(pos_to_player[pos], visited_positions):
                            pos_to_player[pos] = player
                            return True
                return False

            # 各選手から増加路を探す
            for player in candidate_players:
                visited = set()
                dfs(player, visited)
            
            # 全てのポジションが埋まれば完全マッチング成功
            if len(pos_to_player) == len(target_positions):
                return {v: k for k, v in pos_to_player.items()} # player -> pos
            return {}

        # 選手がポジション数より多い場合は、全組み合わせを試す
        if len(players) > len(positions):
            for player_subset in itertools.combinations(players, len(positions)):
                result = find_perfect_matching(list(player_subset), positions, player_positions)
                if result:
                    return result
            return {}
        
        # 選手数とポジション数が同じ場合
        return find_perfect_matching(players, positions, player_positions)