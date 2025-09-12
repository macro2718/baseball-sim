from main_code.config import Positions

class Team:
    def __init__(self, name):
        self.name = name
        self.lineup = []  # 打順の選手リスト（9人）
        self.bench = []   # ベンチの選手リスト
        self.pitchers = []  # 投手リスト
        self.current_pitcher = None
        self.current_batter_index = 0
        
        # 一度退いた選手のリスト（再出場不可）
        self.ejected_players = []
        
        # 守備位置管理（定数を使用）
        self.defensive_positions = {
            Positions.CATCHER: None,
            Positions.FIRST_BASE: None,
            Positions.SECOND_BASE: None,
            Positions.THIRD_BASE: None,
            Positions.SHORTSTOP: None,
            Positions.LEFT_FIELD: None,
            Positions.CENTER_FIELD: None,
            Positions.RIGHT_FIELD: None,
            Positions.DESIGNATED_HITTER: None
        }
        
        # 必須ポジション（定数を使用）
        self.required_positions = Positions.REQUIRED_POSITIONS

    def add_player_to_lineup(self, player, position):
        """選手をラインナップに追加し、守備位置を設定"""
        # ラインナップが満員でないかチェック
        if len(self.lineup) >= 9:
            return False
        
        # 適性チェック
        if not self.can_assign_position(player, position):
            return False
        
        # 選手をラインナップに追加
        player.current_position = position
        self.lineup.append(player)
        self.defensive_positions[position] = player
        return True
    
    def can_assign_position(self, player, position):
        """選手が指定されたポジションに配置可能かチェック"""
        # 有効なポジションかチェック
        if position not in self.defensive_positions:
            return False
        
        # ポジションが空いているかチェック
        if self.defensive_positions[position] is not None:
            return False
        
        # 選手がそのポジションを守れるかチェック
        if hasattr(player, 'can_play_position'):
            return player.can_play_position(position)
        else:
            # 後方互換性のため
            return True
    
    def is_lineup_complete(self):
        """ラインナップが完成しているかチェック（9人すべてのポジションが埋まっているか）"""
        if len(self.lineup) != 9:
            return False
        
        for position in self.required_positions:
            if self.defensive_positions[position] is None:
                return False
        
        return True
    
    def get_missing_positions(self):
        """未配置のポジションリストを取得"""
        missing = []
        for position in self.required_positions:
            if self.defensive_positions[position] is None:
                missing.append(position)
        return missing

    def add_player_to_bench(self, player):
        self.bench.append(player)

    def add_pitcher(self, pitcher):
        self.pitchers.append(pitcher)
        if not self.current_pitcher:
            self.current_pitcher = pitcher

    def next_batter(self):
        if len(self.lineup) == 0:
            raise ValueError("No players in lineup")
        batter = self.lineup[self.current_batter_index]
        self.current_batter_index = (self.current_batter_index + 1) % len(self.lineup)
        return batter
    
    def pinch_hit(self, lineup_index, substitute_player):
        """ベンチから代打を送る（基本機能のみ）"""
        if not (0 <= lineup_index < len(self.lineup)):
            return False, "Invalid lineup index"
        
        if substitute_player not in self.bench:
            return False, f"{substitute_player.name} is not available on the bench"
        
        original_player = self.lineup[lineup_index]
        
        # 代打選手が既に退場していないかチェック
        if substitute_player in self.ejected_players:
            return False, f"{substitute_player.name} has already been ejected and cannot re-enter"
        
        # 代打を実行
        position = original_player.current_position
        
        # 元の選手を退場リストに追加
        self.ejected_players.append(original_player)
        
        # ラインナップを更新
        self.lineup[lineup_index] = substitute_player
        substitute_player.current_position = position
        
        # 守備位置を更新
        self.defensive_positions[position] = substitute_player
        
        # ベンチから代打選手を削除
        self.bench.remove(substitute_player)
        
        return True, f"{substitute_player.name} pinch hits for {original_player.name}"
    
    def change_pitcher(self, new_pitcher):
        """投手交代"""
        if new_pitcher not in self.pitchers:
            return False, f"{new_pitcher.name} is not available as a pitcher"
        
        old_pitcher = self.current_pitcher
        
        # 新しい投手が既に退場していないかチェック
        if new_pitcher in self.ejected_players:
            return False, f"{new_pitcher.name} has already been ejected and cannot re-enter"
        
        # 投手交代を実行
        self.current_pitcher = new_pitcher
        
        # 古い投手を退場リストに追加
        if old_pitcher:
            self.ejected_players.append(old_pitcher)
        
        # 投手リストから新しい投手を削除
        self.pitchers.remove(new_pitcher)
        
        return True, f"{new_pitcher.name} replaces {old_pitcher.name if old_pitcher else 'starting pitcher'}"
        
    def switch_positions(self, player1_index, player2_index):
        # 守備位置交代 (出場メンバーはそのまま)
        if (0 <= player1_index < len(self.lineup) and 
            0 <= player2_index < len(self.lineup) and 
            player1_index != player2_index):
            
            player1 = self.lineup[player1_index]
            player2 = self.lineup[player2_index]
            
            # DHとの交代は制限
            if player1.current_position == "DH" or player2.current_position == "DH":
                return False
            
            # 選手が相手のポジションを守れるかチェック
            if (not player1.can_play_position(player2.current_position) or 
                not player2.can_play_position(player1.current_position)):
                return False
                
            # ポジションを交換
            pos1 = player1.current_position
            pos2 = player2.current_position
            
            # 守備位置辞書を更新
            self.defensive_positions[pos1] = player2
            self.defensive_positions[pos2] = player1
            
            # 選手のポジションを更新
            player1.current_position = pos2
            player2.current_position = pos1
            
            return True
        return False
    
    def set_player_position(self, player_index, new_position):
        """選手の守備位置を変更"""
        if not (0 <= player_index < len(self.lineup)):
            return False, "Invalid player index"
        
        player = self.lineup[player_index]
        old_position = player.current_position
        
        # 新しいポジションが有効かチェック
        if new_position not in self.defensive_positions:
            return False, f"Invalid position: {new_position}"
        
        # 選手がそのポジションを守れるかチェック
        if not player.can_play_position(new_position):
            return False, f"{player.name} cannot play {new_position}"
        
        # 新しいポジションが空いているかチェック
        if self.defensive_positions[new_position] is not None:
            # 既に他の選手がいる場合は交換
            other_player = self.defensive_positions[new_position]
            if not other_player.can_play_position(old_position):
                return False, f"{other_player.name} cannot play {old_position}"
            
            # ポジション交換
            self.defensive_positions[old_position] = other_player
            other_player.current_position = old_position
        else:
            # 元のポジションを空にする
            self.defensive_positions[old_position] = None
        
        # 新しいポジションに配置
        self.defensive_positions[new_position] = player
        player.current_position = new_position
        
        return True, f"{player.name} moved to {new_position}"
    
    def get_defensive_formation(self):
        """現在の守備陣形を取得"""
        formation = {}
        for position, player in self.defensive_positions.items():
            if player is not None:
                formation[position] = player
        return formation
    
    def display_lineup(self):
        """現在のラインナップを表示"""
        print(f"\n=== {self.name} Lineup ===")
        for i, player in enumerate(self.lineup, 1):
            print(f"{i}. {player.name} ({player.current_position})")
        
        missing = self.get_missing_positions()
        if missing:
            print(f"\nMissing positions: {', '.join(missing)}")
    
    def validate_lineup(self):
        """ラインナップの妥当性をチェック"""
        errors = []
        
        # 9人いるかチェック
        if len(self.lineup) != 9:
            errors.append(f"Lineup has {len(self.lineup)} players, need 9")
        
        # 全ポジションが埋まっているかチェック
        missing_positions = self.get_missing_positions()
        if missing_positions:
            errors.append(f"Missing positions: {', '.join(missing_positions)}")
        
        # 重複ポジションチェック
        used_positions = []
        for player in self.lineup:
            if player.current_position in used_positions:
                errors.append(f"Duplicate position: {player.current_position}")
            used_positions.append(player.current_position)
        
        # 適性チェック
        for player in self.lineup:
            if not player.can_play_position(player.current_position):
                errors.append(f"{player.name} cannot play {player.current_position}")
        
        return errors
    
    def clear_lineup(self):
        """ラインナップをリセット"""
        # 全選手をベンチに移動
        for player in self.lineup:
            player.current_position = None
            self.bench.append(player)
        
        self.lineup.clear()
        self.defensive_positions = {pos: None for pos in self.required_positions}
        self.current_batter_index = 0
        
        # 退場リストもリセット（新しい試合の場合）
        self.ejected_players.clear()
    
    def display_ejected_players(self):
        """退場した選手のリストを表示"""
        if self.ejected_players:
            print(f"\n=== {self.name} 退場選手 ===")
            for i, player in enumerate(self.ejected_players, 1):
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                print(f"{i}. {player.name} ({primary_pos})")
        else:
            print(f"\n{self.name}: 退場選手なし")
    
    def game_summary(self):
        """試合終了時のサマリー表示"""
        print(f"\n=== {self.name} 試合サマリー ===")
        print(f"最終ラインナップ:")
        for i, player in enumerate(self.lineup, 1):
            print(f"  {i}. {player.name} ({player.current_position})")
        
        print(f"現在の投手: {self.current_pitcher.name if self.current_pitcher else 'なし'}")
        
        if self.ejected_players:
            print(f"退場選手 ({len(self.ejected_players)}人):")
            for player in self.ejected_players:
                print(f"  - {player.name}")
        else:
            print("退場選手: なし")
    
    def substitute_player(self, lineup_index, substitute_player):
        """守備交代（代走・守備固めなど）- 選手オブジェクト版"""
        if not (0 <= lineup_index < len(self.lineup)):
            return False, "Invalid lineup index"
        
        if substitute_player not in self.bench:
            return False, f"{substitute_player.name} is not available on the bench"
        
        original_player = self.lineup[lineup_index]
        
        # 代替選手が既に退場していないかチェック
        if substitute_player in self.ejected_players:
            return False, f"{substitute_player.name} has already been ejected and cannot re-enter"
        
        # 代替選手がそのポジションを守れるかチェック
        position = original_player.current_position
        if not substitute_player.can_play_position(position):
            return False, f"{substitute_player.name} cannot play {position}"
        
        # 交代を実行
        # 元の選手を退場リストに追加
        self.ejected_players.append(original_player)
        
        # ラインナップを更新
        self.lineup[lineup_index] = substitute_player
        substitute_player.current_position = position
        
        # 守備位置を更新
        self.defensive_positions[position] = substitute_player
        
        # ベンチから代替選手を削除
        self.bench.remove(substitute_player)
        
        return True, f"{substitute_player.name} substitutes for {original_player.name} at {position}"
    
    def substitute_player_by_index(self, bench_player_index, lineup_index):
        """守備交代（代走・守備固めなど）- インデックス版"""
        if not (0 <= bench_player_index < len(self.bench) and 0 <= lineup_index < len(self.lineup)):
            return False, "Invalid player index"
        
        substitute_player = self.bench[bench_player_index]
        return self.substitute_player(lineup_index, substitute_player)
    
    def is_player_eligible(self, player):
        """選手が出場可能かチェック（退場していないか）"""
        return player not in self.ejected_players
    
    def get_available_bench_players(self):
        """出場可能なベンチ選手のリストを取得"""
        return [player for player in self.bench if self.is_player_eligible(player)]
    
    def get_available_pitchers(self):
        """出場可能な投手のリストを取得"""
        return [pitcher for pitcher in self.pitchers if self.is_player_eligible(pitcher)]
    
    def validate_defensive_positions(self):
        """守備位置の妥当性をチェックし、問題があれば警告を返す"""
        warnings = []
        errors = []
        
        # 必須ポジションの重複チェック
        assigned_positions = {}
        for position, player in self.defensive_positions.items():
            if player is not None:
                if position in assigned_positions:
                    errors.append(f"Position {position} is assigned to multiple players")
                assigned_positions[position] = player
        
        # 全ての必須ポジションが埋まっているかチェック
        missing_positions = []
        for position in self.required_positions:
            if position not in assigned_positions:
                missing_positions.append(position)
        
        if missing_positions:
            errors.append(f"Missing required positions: {', '.join(missing_positions)}")
        
        # 各選手が適切なポジションにいるかチェック
        for position, player in assigned_positions.items():
            if hasattr(player, 'can_play_position'):
                if not player.can_play_position(position):
                    warnings.append(f"{player.name} may not be suitable for position {position}")
            else:
                # 後方互換性のための基本的なチェック
                if hasattr(player, 'eligible_positions'):
                    if position not in player.eligible_positions:
                        warnings.append(f"{player.name} is not in their natural position ({position})")
        
        # ラインナップの人数チェック
        if len(self.lineup) != 9:
            errors.append(f"Lineup must have exactly 9 players, currently has {len(self.lineup)}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0
        }
    
    def check_defensive_readiness(self):
        """攻守交代時の守備準備状況をチェック"""
        validation_result = self.validate_defensive_positions()
        
        if not validation_result['is_valid']:
            return False, validation_result['errors']
        
        # 投手がセットされているかチェック
        if self.current_pitcher is None:
            validation_result['errors'].append("No pitcher assigned")
            return False, validation_result['errors']
        
        return True, validation_result['warnings']
