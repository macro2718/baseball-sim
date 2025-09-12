"""
Strategy menu functionality for the Baseball GUI
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# プロジェクト設定を使用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_code.config import setup_project_environment
setup_project_environment()

from main_code.core.lineup_manager import LineupManager
from main_code.core.substitution_manager import SubstitutionManager
from .gui_defense_mode import DefenseChangeMode

class StrategyManager:
    def __init__(self, root, text, main_gui=None):
        self.root = root
        self.text = text
        self.game_state = None
        self.main_gui = main_gui
    
    def set_game_state(self, game_state):
        """ゲーム状態を設定"""
        self.game_state = game_state
    
    def _is_game_over(self):
        """ゲームが終了しているかチェック"""
        if not self.game_state:
            return True
        
        return self.game_state.game_ended
    
    def show_offense_menu(self):
        """攻撃側の采配メニューを表示"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        if not self.game_state or not self.game_state.batting_team:
            return
        
        offense_window = tk.Toplevel(self.root)
        offense_window.title(self.text["offense_strategy"])
        offense_window.geometry("650x780")  # 500x600 * 1.3
        
        # 選手リスト
        team = self.game_state.batting_team
        substitution_manager = SubstitutionManager(team)
        
        lineup_frame = ttk.LabelFrame(offense_window, text=self.text["batting_order"])
        lineup_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        formatted_lineup = substitution_manager.get_formatted_lineup()
        for i, info in enumerate(formatted_lineup):
            player_frame = ttk.Frame(lineup_frame)
            player_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(player_frame, text=info).pack(side=tk.LEFT, padx=5)
            
            if i == team.current_batter_index:
                ttk.Label(player_frame, text=self.text["current_batter"]).pack(side=tk.RIGHT, padx=5)
        
        # ベンチの選手リスト（退場していない選手のみ）
        bench_frame = ttk.LabelFrame(offense_window, text=self.text["bench"])
        bench_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        formatted_bench = substitution_manager.get_formatted_bench()
        for i, info in enumerate(formatted_bench):
            player_frame = ttk.Frame(bench_frame)
            player_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(player_frame, text=info).pack(side=tk.LEFT, padx=5)
            ttk.Button(player_frame, text=self.text["pinch_hit"], 
                      command=lambda idx=i, sm=substitution_manager: self._pinch_hit_unified(idx, sm)).pack(side=tk.RIGHT, padx=5)
        
        # 閉じるボタン
        ttk.Button(offense_window, text=self.text["close"], command=offense_window.destroy).pack(pady=10)
    
    def show_defense_menu(self):
        """守備側の采配メニューを表示"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        if not self.game_state or not self.game_state.fielding_team:
            return
        
        defense_window = tk.Toplevel(self.root)
        defense_window.title(self.text["defense_strategy"])
        defense_window.geometry("700x900")
        
        # 投手交代セクション
        team = self.game_state.fielding_team
        substitution_manager = SubstitutionManager(team)
        
        pitchers_frame = ttk.LabelFrame(defense_window, text=self.text["pitchers"])
        pitchers_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 現在の投手表示
        current_pitcher_frame = ttk.Frame(pitchers_frame)
        current_pitcher_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(current_pitcher_frame, text=self.text["current_pitcher"].format(team.current_pitcher)).pack(pady=5)
        
        # 控え投手リスト（統一化された処理を使用）
        formatted_pitchers = substitution_manager.get_formatted_pitchers()
        for i, info in enumerate(formatted_pitchers):
            pitcher_frame = ttk.Frame(pitchers_frame)
            pitcher_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(pitcher_frame, text=info).pack(side=tk.LEFT, padx=5)
            ttk.Button(pitcher_frame, text=self.text["change"], 
                      command=lambda idx=i, sm=substitution_manager: self._change_pitcher_unified(idx, sm)).pack(side=tk.RIGHT, padx=5)
        
        # 新しい守備変更モードシステム
        defense_change_frame = ttk.LabelFrame(defense_window, text="Defensive Changes")
        defense_change_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 新しい守備変更モードボタン
        new_defense_mode_button = ttk.Button(defense_change_frame, text="Defense Change Mode", 
                                           command=lambda sm=substitution_manager: self._start_defense_change_mode(sm))
        new_defense_mode_button.pack(fill=tk.X, padx=10, pady=10)
        
        # 説明テキスト
        mode_instruction = ("Defense Change Mode allows you to:\n"
                          "• Make multiple position changes and substitutions\n"
                          "• Temporarily have invalid formations during changes\n"
                          "• Apply all changes at once when complete\n"
                          "• Intuitive drag-and-drop style interface")
        ttk.Label(defense_change_frame, text=mode_instruction, 
                 wraplength=500, justify="left", font=("Arial", 9)).pack(padx=10, pady=5)
        
        # 守備陣形表示ボタン
        formation_button = ttk.Button(defense_change_frame, text="View Current Defensive Formation", 
                                    command=self._show_defensive_formation)
        formation_button.pack(fill=tk.X, padx=10, pady=5)
        
        # 閉じるボタン
        ttk.Button(defense_window, text=self.text["close"], command=defense_window.destroy).pack(pady=10)
    
    # 統一化されたメソッド群
    def _pinch_hit_unified(self, bench_index, substitution_manager):
        """統一化された代打処理"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
        
        try:
            lineup_index = substitution_manager.team.current_batter_index
            success, message = substitution_manager.execute_pinch_hit(bench_index, lineup_index)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Pinch Hit", f"Pinch hit completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Pinch hit error: {str(e)}")
    
    def _change_pitcher_unified(self, pitcher_index, substitution_manager):
        """統一化された投手交代処理"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
        
        try:
            success, message = substitution_manager.execute_pitcher_change(pitcher_index)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Pitcher Change", f"Pitcher change completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Pitcher change error: {str(e)}")
    
    def _show_unified_defense_dialog(self, substitution_manager):
        """統一化された守備交代ダイアログ - ステップ1: 交代させる選手を選択"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Defensive Changes - Step 1: Select Player to Change")
        dialog.geometry("600x500")
        
        ttk.Label(dialog, text="Select the player you want to change:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 現在の守備陣リスト
        fielders_frame = ttk.Frame(dialog)
        fielders_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        formatted_lineup = substitution_manager.get_formatted_lineup()
        for i, info in enumerate(formatted_lineup):
            player_frame = ttk.Frame(fielders_frame)
            player_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(player_frame, text=info).pack(side=tk.LEFT, padx=5)
            
            # 選手交代ボタン
            substitute_button = ttk.Button(player_frame, text="Substitute", 
                                         command=lambda idx=i: [self._show_substitution_step2_dialog(dialog, idx, substitution_manager)])
            substitute_button.pack(side=tk.RIGHT, padx=2)
            
            # ポジション変更ボタン（DHは除外）
            if substitution_manager.team.lineup[i].current_position != "DH":
                position_button = ttk.Button(player_frame, text="Change Position", 
                                           command=lambda idx=i: [self._show_position_change_dialog(dialog, idx, substitution_manager)])
                position_button.pack(side=tk.RIGHT, padx=2)
        
        # 閉じるボタン
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def _show_substitution_step2_dialog(self, parent_dialog, fielder_idx, substitution_manager):
        """守備交代 - ステップ2: ベンチ選手を選択（統一化）"""
        parent_dialog.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Substitute - Step 2: Select Bench Player")
        dialog.geometry("600x500")
        
        # 交代する選手の情報を表示
        fielder_to_replace = substitution_manager.team.lineup[fielder_idx]
        ttk.Label(dialog, text=f"Replacing: {substitution_manager.format_player_info(fielder_to_replace)}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # ベンチ選手リスト
        bench_frame = ttk.Frame(dialog)
        bench_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        available_bench = substitution_manager.get_available_bench_players()
        if not available_bench:
            ttk.Label(bench_frame, text="No available bench players").pack(pady=20)
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            return
        
        current_pos = fielder_to_replace.current_position
        
        bench_count = 0
        for i, player in enumerate(available_bench):
            # ポジション適性チェック
            can_play = player.can_play_position(current_pos)
            status = "✓" if can_play else "✗"
            
            player_frame = ttk.Frame(bench_frame)
            player_frame.pack(fill=tk.X, pady=2)
            
            player_info = f"{bench_count+1}. {substitution_manager.format_player_info(player)} {status}"
            
            if can_play:
                player_button = ttk.Button(player_frame, text=player_info,
                                         command=lambda idx=i: [self._execute_defense_sub_unified(idx, fielder_idx, substitution_manager), dialog.destroy()])
                player_button.pack(fill=tk.X, padx=5)
            else:
                ttk.Label(player_frame, text=player_info, foreground="gray").pack(side=tk.LEFT, padx=5)
                ttk.Label(player_frame, text="Cannot play this position", foreground="red").pack(side=tk.RIGHT, padx=5)
            
            bench_count += 1
        
        # 閉じるボタン
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)
    
    def _show_position_change_dialog(self, parent_dialog, player_idx, substitution_manager):
        """ポジション変更ダイアログ（統一化）"""
        parent_dialog.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Position")
        dialog.geometry("500x400")
        
        player = substitution_manager.team.lineup[player_idx]
        ttk.Label(dialog, text=f"Change position for: {substitution_manager.format_player_info(player)}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # 他の選手とのポジション交換オプション
        exchange_frame = ttk.LabelFrame(dialog, text="Exchange Position with Another Player")
        exchange_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, other_player in enumerate(substitution_manager.team.lineup):
            if i != player_idx and other_player.current_position != "DH":
                # 相互適性チェック
                can_exchange = (player.can_play_position(other_player.current_position) and 
                              other_player.can_play_position(player.current_position))
                
                if can_exchange:
                    exchange_button = ttk.Button(exchange_frame, 
                                               text=f"Exchange with {other_player.name} ({other_player.current_position})",
                                               command=lambda idx1=player_idx, idx2=i: [self._execute_position_exchange_unified(idx1, idx2, substitution_manager), dialog.destroy()])
                    exchange_button.pack(fill=tk.X, padx=5, pady=2)
        
        # 閉じるボタン
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)
    
    def _execute_defense_sub_unified(self, bench_index, lineup_index, substitution_manager):
        """統一化された守備固めを実行"""
        try:
            success, message = substitution_manager.execute_defensive_substitution(bench_index, lineup_index)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Defensive Substitution", f"Substitution completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Defensive substitution error: {str(e)}")
    
    def _execute_position_exchange_unified(self, player1_idx, player2_idx, substitution_manager):
        """統一化された2人の選手のポジションを交換"""
        try:
            success, message = substitution_manager.execute_position_switch(player1_idx, player2_idx)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Position Exchange", f"Position exchange completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Position exchange error: {str(e)}")

    # 既存のメソッド群（後方互換性のため残す）
    
    def _execute_individual_position_change(self, player_idx, new_position):
        """個別選手のポジション変更を実行"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        team = self.game_state.fielding_team
        player = team.lineup[player_idx]
        
        # 現在の守備位置を取得
        current_pos = player.current_position if hasattr(player, 'current_position') and player.current_position else player.position
        
        # 同じポジションの場合は変更不要
        if current_pos == new_position:
            messagebox.showinfo("No Change", f"{player.name} is already playing {new_position}")
            return
        
        # ポジション適性をチェック
        if not player.can_play_position(new_position):
            messagebox.showwarning("Warning", f"{player.name} cannot play {new_position}")
            return
        
        # 新しいポジションに既に他の選手がいるかチェック
        existing_player = None
        for p in team.lineup:
            existing_pos = p.current_position if hasattr(p, 'current_position') and p.current_position else p.position
            if existing_pos == new_position and p != player:
                existing_player = p
                break
        
        if existing_player:
            # 既存の選手が現在の選手のポジションを守れるかチェック
            if not existing_player.can_play_position(current_pos):
                messagebox.showwarning("Warning", 
                    f"Position change not possible:\n{existing_player.name} cannot play {current_pos}")
                return
            
            # 確認ダイアログを表示
            confirm = messagebox.askyesno("Position Exchange", 
                f"Position change will cause:\n"
                f"{player.name}: {current_pos} → {new_position}\n"
                f"{existing_player.name}: {new_position} → {current_pos}\n\n"
                f"Do you want to proceed?")
            
            if not confirm:
                return
            
            # ポジションを交換
            if hasattr(existing_player, 'current_position'):
                existing_player.current_position = current_pos
            else:
                existing_player.position = current_pos
        
        # 選手のポジションを変更
        if hasattr(player, 'current_position'):
            player.current_position = new_position
        else:
            player.position = new_position
        
        # メインGUIの表示を更新
        if self.main_gui:
            self.main_gui._update_scoreboard(self.game_state)
        
        if existing_player:
            messagebox.showinfo("Position Change", 
                f"Position exchange completed:\n"
                f"{player.name}: {current_pos} → {new_position}\n"
                f"{existing_player.name}: {new_position} → {current_pos}")
        else:
            messagebox.showinfo("Position Change", 
                f"Position change completed:\n{player.name}: {current_pos} → {new_position}")
    
    def _pinch_hit(self, bench_index):
        """代打を送る"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        if not self.game_state:
            return
        
        try:
            batting_team = self.game_state.batting_team
            current_batter_idx = batting_team.current_batter_index
            available_bench = batting_team.get_available_bench_players()
            
            if bench_index >= len(available_bench):
                messagebox.showerror(self.text["error"], "Selected player is not available")
                return
                
            pinch_hitter = available_bench[bench_index]
            
            # 新しい代打システムを使用
            success, message = batting_team.pinch_hit(current_batter_idx, pinch_hitter)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                
                messagebox.showinfo("Pinch Hit", f"Pinch hit completed:\n{message}")
            else:
                messagebox.showerror(self.text["error"], message)
                
        except (IndexError, AttributeError) as e:
            messagebox.showerror(self.text["error"], f"Pinch hit error: {str(e)}")
    
    def _change_pitcher(self, pitcher_index):
        """投手交代"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        if not self.game_state:
            return
        
        try:
            fielding_team = self.game_state.fielding_team
            available_pitchers = fielding_team.get_available_pitchers()
            
            if pitcher_index >= len(available_pitchers):
                messagebox.showerror(self.text["error"], "Selected pitcher is not available")
                return
                
            old_pitcher = fielding_team.current_pitcher
            new_pitcher = available_pitchers[pitcher_index]
            
            # 新しい投手交代システムを使用
            success, message = fielding_team.change_pitcher(new_pitcher)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                
                messagebox.showinfo("Pitcher Change", f"Pitcher change completed:\n{old_pitcher.name} → {new_pitcher.name}")
            else:
                messagebox.showerror(self.text["error"], message)
                
        except (IndexError, AttributeError) as e:
            messagebox.showerror(self.text["error"], f"Pitcher change error: {str(e)}")
    
    def _defense_substitution(self, bench_index):
        """守備固め"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        if not self.game_state:
            return
            
        team = self.game_state.fielding_team
        
        # 交代させる選手を選択するダイアログ
        dialog = tk.Toplevel(self.root)
        dialog.title(self.text["select_player"])
        dialog.geometry("390x390")  # 300x300 * 1.3
        
        ttk.Label(dialog, text=self.text["select_player"]).pack(pady=10)
        
        lineup_frame = ttk.Frame(dialog)
        lineup_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, player in enumerate(team.lineup):
            player_button = ttk.Button(lineup_frame, text=f"{i+1}. {player}", 
                                      command=lambda idx=i: [self._execute_defense_sub(bench_index, idx), dialog.destroy()])
            player_button.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(dialog, text=self.text["cancel"], command=dialog.destroy).pack(pady=10)
    
    def _execute_defense_sub(self, bench_index, lineup_index):
        """守備固めを実行"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        try:
            team = self.game_state.fielding_team
            available_bench = team.get_available_bench_players()
            
            if bench_index >= len(available_bench):
                messagebox.showerror("Error", "Selected player is not available")
                return
                
            substitute = available_bench[bench_index]
            original_player = team.lineup[lineup_index]
            
            # 守備位置を取得
            position = original_player.current_position if hasattr(original_player, 'current_position') and original_player.current_position else original_player.position
            
            # ポジション適性をチェック
            if not substitute.can_play_position(position):
                messagebox.showwarning("Warning", f"{substitute.name} cannot play {position}")
                return
            
            # 新しい選手交代システムを使用
            success, message = team.substitute_player(lineup_index, substitute)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                
                messagebox.showinfo("Defensive Substitution", 
                                  f"Substitution completed:\n{original_player.name} → {substitute.name}\nPosition: {position}")
            else:
                messagebox.showerror("Error", message)
                
        except (IndexError, AttributeError) as e:
            messagebox.showerror("Error", f"Defense substitution error: {str(e)}")

    def _change_pitcher_unified(self, pitcher_index, substitution_manager):
        """統一化された投手交代処理"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
        
        try:
            success, message = substitution_manager.execute_pitcher_change(pitcher_index)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Pitcher Change", f"Pitcher change completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Pitcher change error: {str(e)}")
    
    def _show_unified_defense_dialog(self, substitution_manager):
        """統一化された守備交代ダイアログ - ステップ1: 交代させる選手を選択"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Defensive Changes - Step 1: Select Player to Change")
        dialog.geometry("600x500")
        
        ttk.Label(dialog, text="Select the player you want to change:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 現在の守備陣リスト
        fielders_frame = ttk.Frame(dialog)
        fielders_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        formatted_lineup = substitution_manager.get_formatted_lineup()
        for i, info in enumerate(formatted_lineup):
            player_frame = ttk.Frame(fielders_frame)
            player_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(player_frame, text=info).pack(side=tk.LEFT, padx=5)
            
            # 選手交代ボタン
            substitute_button = ttk.Button(player_frame, text="Substitute", 
                                         command=lambda idx=i: [self._show_substitution_step2_dialog_unified(dialog, idx, substitution_manager)])
            substitute_button.pack(side=tk.RIGHT, padx=2)
            
            # ポジション変更ボタン（DHは除外）
            if substitution_manager.team.lineup[i].current_position != "DH":
                position_button = ttk.Button(player_frame, text="Change Position", 
                                           command=lambda idx=i: [self._show_position_change_dialog_unified(dialog, idx, substitution_manager)])
                position_button.pack(side=tk.RIGHT, padx=2)
        
        # 閉じるボタン
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def _show_substitution_step2_dialog_unified(self, parent_dialog, fielder_idx, substitution_manager):
        """守備交代 - ステップ2: ベンチ選手を選択（統一化）"""
        parent_dialog.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Substitute - Step 2: Select Bench Player")
        dialog.geometry("600x500")
        
        # 交代する選手の情報を表示
        fielder_to_replace = substitution_manager.team.lineup[fielder_idx]
        ttk.Label(dialog, text=f"Replacing: {substitution_manager.format_player_info(fielder_to_replace)}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # ベンチ選手リスト
        bench_frame = ttk.Frame(dialog)
        bench_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        available_bench = substitution_manager.get_available_bench_players()
        if not available_bench:
            ttk.Label(bench_frame, text="No available bench players").pack(pady=20)
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            return
        
        current_pos = fielder_to_replace.current_position
        
        bench_count = 0
        for i, player in enumerate(available_bench):
            # ポジション適性チェック
            can_play = player.can_play_position(current_pos)
            status = "✓" if can_play else "✗"
            
            player_frame = ttk.Frame(bench_frame)
            player_frame.pack(fill=tk.X, pady=2)
            
            player_info = f"{bench_count+1}. {substitution_manager.format_player_info(player)} {status}"
            
            if can_play:
                player_button = ttk.Button(player_frame, text=player_info,
                                         command=lambda idx=i: [self._execute_defense_sub_unified(idx, fielder_idx, substitution_manager), dialog.destroy()])
                player_button.pack(fill=tk.X, padx=5)
            else:
                ttk.Label(player_frame, text=player_info, foreground="gray").pack(side=tk.LEFT, padx=5)
                ttk.Label(player_frame, text="Cannot play this position", foreground="red").pack(side=tk.RIGHT, padx=5)
            
            bench_count += 1
        
        # 閉じるボタン
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)
    
    def _show_position_change_dialog_unified(self, parent_dialog, player_idx, substitution_manager):
        """ポジション変更ダイアログ（統一化）"""
        parent_dialog.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Position")
        dialog.geometry("500x400")
        
        player = substitution_manager.team.lineup[player_idx]
        ttk.Label(dialog, text=f"Change position for: {substitution_manager.format_player_info(player)}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # 他の選手とのポジション交換オプション
        exchange_frame = ttk.LabelFrame(dialog, text="Exchange Position with Another Player")
        exchange_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, other_player in enumerate(substitution_manager.team.lineup):
            if i != player_idx and other_player.current_position != "DH":
                # 相互適性チェック
                can_exchange = (player.can_play_position(other_player.current_position) and 
                              other_player.can_play_position(player.current_position))
                
                if can_exchange:
                    exchange_button = ttk.Button(exchange_frame, 
                                               text=f"Exchange with {other_player.name} ({other_player.current_position})",
                                               command=lambda idx1=player_idx, idx2=i: [self._execute_position_exchange_unified(idx1, idx2, substitution_manager), dialog.destroy()])
                    exchange_button.pack(fill=tk.X, padx=5, pady=2)
        
        # 閉じるボタン
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)
    
    def _execute_defense_sub_unified(self, bench_index, lineup_index, substitution_manager):
        """統一化された守備固めを実行"""
        try:
            success, message = substitution_manager.execute_defensive_substitution(bench_index, lineup_index)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Defensive Substitution", f"Substitution completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Defensive substitution error: {str(e)}")
    
    def _execute_position_exchange_unified(self, player1_idx, player2_idx, substitution_manager):
        """統一化された2人の選手のポジションを交換"""
        try:
            success, message = substitution_manager.execute_position_switch(player1_idx, player2_idx)
            
            if success:
                # メインGUIの表示を更新
                if self.main_gui:
                    self.main_gui._update_scoreboard(self.game_state)
                messagebox.showinfo("Position Exchange", f"Position exchange completed:\n{message}")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Position exchange error: {str(e)}")
            
    def _show_defensive_formation(self):
        """守備陣形を表示するダイアログ"""
        if not self.game_state:
            return
            
        team = self.game_state.fielding_team
        
        formation_window = tk.Toplevel(self.root)
        formation_window.title(f"{team.name} - Defensive Formation")
        formation_window.geometry("520x650")  # 400x500 * 1.3
        
        # 守備陣形表示
        ttk.Label(formation_window, text=f"{team.name} Defensive Formation", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # ポジション順で表示
        position_order = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
        
        formation_frame = ttk.Frame(formation_window)
        formation_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for pos in position_order:
            pos_frame = ttk.Frame(formation_frame)
            pos_frame.pack(fill=tk.X, pady=3)
            
            # ポジションの選手を見つける
            player = None
            batting_order = "-"
            for i, p in enumerate(team.lineup):
                if p.position == pos:
                    player = p
                    batting_order = str(i + 1)
                    break
            
            if player:
                player_info = f"{player.name} (Fielding: {player.fielding_skill})"
            else:
                player_info = "Empty"
                
            ttk.Label(pos_frame, text=f"{pos}:", width=4, anchor="e").pack(side=tk.LEFT)
            ttk.Label(pos_frame, text=player_info, width=20, anchor="w").pack(side=tk.LEFT, padx=10)
            ttk.Label(pos_frame, text=f"Order: {batting_order}", width=10, anchor="w").pack(side=tk.LEFT)
        
        # 投手情報
        pitcher_frame = ttk.LabelFrame(formation_window, text="Pitcher")
        pitcher_frame.pack(fill=tk.X, padx=20, pady=10)
        
        pitcher = team.current_pitcher
        pitcher_info = f"{pitcher.name} ({pitcher.pitcher_type}) - Stamina: {pitcher.stamina}%"
        ttk.Label(pitcher_frame, text=pitcher_info).pack(pady=5)
        
        ttk.Button(formation_window, text="Close", command=formation_window.destroy).pack(pady=10)
     
    
    def _show_lineup_setup_for_team(self, team):
        """指定されたチーム用のスタメン設定ダイアログを表示"""
        if not team:
            messagebox.showerror("Error", "Team is not available")
            return
            
        try:
            lineup_manager = LineupManager(team)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create lineup manager: {e}")
            return
        
        try:
            setup_window = tk.Toplevel(self.root)
            setup_window.title(f"{team.name} - Lineup Setup")
            setup_window.geometry("780x910")  # 600x700 * 1.3
            
            # メインフレーム
            main_frame = ttk.Frame(setup_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 現在のラインナップ表示
            lineup_frame = ttk.LabelFrame(main_frame, text="Current Lineup")
            lineup_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # スクロール可能なフレーム
            canvas = tk.Canvas(lineup_frame)
            scrollbar = ttk.Scrollbar(lineup_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # ラインナップ表示を更新する関数
            def refresh_lineup_display():
                for widget in scrollable_frame.winfo_children():
                    widget.destroy()
                    
                # ヘッダー
                header_frame = ttk.Frame(scrollable_frame)
                header_frame.pack(fill=tk.X, pady=2)
                ttk.Label(header_frame, text="Order", width=8, anchor="center").pack(side=tk.LEFT)
                ttk.Label(header_frame, text="Player", width=15, anchor="center").pack(side=tk.LEFT)
                ttk.Label(header_frame, text="Position", width=8, anchor="center").pack(side=tk.LEFT)
                ttk.Label(header_frame, text="Actions", width=20, anchor="center").pack(side=tk.LEFT)
                
                # 選手リスト
                for i, player in enumerate(team.lineup):
                    player_frame = ttk.Frame(scrollable_frame)
                    player_frame.pack(fill=tk.X, pady=1)
                    
                    ttk.Label(player_frame, text=str(i+1), width=8, anchor="center").pack(side=tk.LEFT)
                    # 選手名の後に適性ポジションを表示
                    primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                    player_display = f"{player.name} ({primary_pos})"
                    ttk.Label(player_frame, text=player_display, width=20, anchor="center").pack(side=tk.LEFT)
                    # 現在の守備位置を表示
                    current_pos = player.current_position or primary_pos
                    ttk.Label(player_frame, text=current_pos, width=8, anchor="center").pack(side=tk.LEFT)
                    
                    # アクションボタン
                    action_frame = ttk.Frame(player_frame)
                    action_frame.pack(side=tk.LEFT, padx=5)
                    
                    ttk.Button(action_frame, text="Change Pos", width=10,
                              command=lambda idx=i: change_position(idx)).pack(side=tk.LEFT, padx=1)
                    
                    order_btn = ttk.Button(action_frame, text="Change Order", width=12,
                                          command=lambda: self._show_batting_order_dialog(team, lineup_manager, refresh_lineup_display))
                    order_btn.pack(side=tk.LEFT, padx=1)
                    order_btn.pack(side=tk.LEFT, padx=1)
                              
            # ポジション変更機能
            def change_position(player_index):
                try:
                    if player_index >= len(team.lineup):
                        messagebox.showerror("Error", f"Invalid player index: {player_index}")
                        return
                    
                    player = team.lineup[player_index]
                    
                    pos_window = tk.Toplevel(setup_window)
                    pos_window.title("Change Position")
                    pos_window.geometry("455x585")  # 350x450 * 1.3
                    
                    primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                    current_pos = player.current_position or primary_pos
                    ttk.Label(pos_window, text=f"Change position for {player.name}").pack(pady=10)
                    ttk.Label(pos_window, text=f"Primary Position: {primary_pos}").pack(pady=2)
                    ttk.Label(pos_window, text=f"Current Position: {current_pos}").pack(pady=2)
                    
                    # 選手の適性ポジションを取得（DHを除外）
                    if hasattr(player, 'get_display_eligible_positions'):
                        eligible_positions = player.get_display_eligible_positions()
                        ttk.Label(pos_window, text=f"Eligible Positions: {', '.join(eligible_positions)}").pack(pady=5)
                    elif hasattr(player, 'eligible_positions'):
                        eligible_positions = [pos for pos in player.eligible_positions if pos != "DH"]
                        ttk.Label(pos_window, text=f"Eligible Positions: {', '.join(eligible_positions)}").pack(pady=5)
                    else:
                        eligible_positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
                    
                    # ポジション選択
                    position_var = tk.StringVar(value=current_pos)
                    
                    # 適性のあるポジションのみ表示
                    for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]:
                        if pos in eligible_positions:
                            ttk.Radiobutton(pos_window, text=f"{pos} ✓", variable=position_var, value=pos).pack(anchor="w", padx=20)
                        else:
                            # 適性のないポジションは無効化
                            rb = ttk.Radiobutton(pos_window, text=f"{pos} ✗ (Not eligible)", variable=position_var, value=pos, state="disabled")
                            rb.pack(anchor="w", padx=20)
                    
                    def apply_position_change():
                        try:
                            new_pos = position_var.get()
                            
                            # 適性チェック
                            if hasattr(player, 'can_play_position') and not player.can_play_position(new_pos):
                                messagebox.showerror("Error", f"{player.name} cannot play position {new_pos}")
                                return
                            
                            success, message = lineup_manager.set_player_position(player_index, new_pos)
                            if success:
                                refresh_lineup_display()
                                pos_window.destroy()
                                messagebox.showinfo("Success", message)
                            else:
                                messagebox.showerror("Error", message)
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to change position: {str(e)}")
                    
                    ttk.Button(pos_window, text="Apply", command=apply_position_change).pack(pady=10)
                    ttk.Button(pos_window, text="Cancel", command=pos_window.destroy).pack(pady=5)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Error", f"Failed to open position change dialog: {str(e)}")
            
            # エラー表示エリア
            error_frame = ttk.LabelFrame(main_frame, text="Lineup Validation")
            error_frame.pack(fill=tk.X, padx=10, pady=5)
            
            validation_text = tk.Text(error_frame, height=4, width=70)
            validation_text.pack(fill=tk.X, padx=5, pady=5)
            
            def validate_and_display():
                """ラインナップをバリデートして結果を表示"""
                errors = lineup_manager.validate_lineup()
                validation_text.delete(1.0, tk.END)
                if errors:
                    validation_text.insert(tk.END, "Lineup Errors:\n")
                    for error in errors:
                        validation_text.insert(tk.END, f"• {error}\n")
                    validation_text.config(fg="red")
                else:
                    validation_text.insert(tk.END, "✓ Lineup is valid!")
                    validation_text.config(fg="green")
            
            # バリデーションボタン
            ttk.Button(error_frame, text="Validate Lineup", command=validate_and_display).pack(pady=5)
            
            # 完了ボタン
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=10)
            ttk.Button(button_frame, text="Complete Setup", command=setup_window.destroy).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Validate & Complete", 
                      command=lambda: [validate_and_display(), setup_window.destroy() if not lineup_manager.validate_lineup() else None]).pack(side=tk.LEFT, padx=5)
            
            # 初期表示
            refresh_lineup_display()
            validate_and_display()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to setup lineup dialog: {e}")
            if 'setup_window' in locals():
                setup_window.destroy()

    def _show_batting_order_dialog(self, team, lineup_manager, refresh_callback):
        """打順変更ダイアログを表示"""
        order_window = tk.Toplevel(self.root)
        order_window.title("Change Batting Order")
        order_window.geometry("500x600")
        
        ttk.Label(order_window, text=f"Change Batting Order for {team.name}", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # 現在の打順を表示
        order_frame = ttk.LabelFrame(order_window, text="Current Batting Order")
        order_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # スクロール可能なフレーム
        canvas = tk.Canvas(order_frame)
        scrollbar = ttk.Scrollbar(order_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 選択された選手を追跡
        selected_player = [None]
        
        def refresh_order_display():
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
                
            # ヘッダー
            header_frame = ttk.Frame(scrollable_frame)
            header_frame.pack(fill=tk.X, pady=2)
            ttk.Label(header_frame, text="Order", width=8, anchor="center").pack(side=tk.LEFT)
            ttk.Label(header_frame, text="Player", width=20, anchor="center").pack(side=tk.LEFT)
            ttk.Label(header_frame, text="Position", width=10, anchor="center").pack(side=tk.LEFT)
            ttk.Label(header_frame, text="Action", width=15, anchor="center").pack(side=tk.LEFT)
            
            # 選手リスト
            for i, player in enumerate(team.lineup):
                player_frame = ttk.Frame(scrollable_frame)
                player_frame.pack(fill=tk.X, pady=1)
                
                # 選択されている選手をハイライト
                bg_color = "lightblue" if selected_player[0] == i else "white"
                
                order_label = ttk.Label(player_frame, text=str(i+1), width=8, anchor="center")
                order_label.pack(side=tk.LEFT)
                
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                current_pos = player.current_position or primary_pos
                player_display = f"{player.name}"
                
                player_label = ttk.Label(player_frame, text=player_display, width=20, anchor="center")
                player_label.pack(side=tk.LEFT)
                
                pos_label = ttk.Label(player_frame, text=current_pos, width=10, anchor="center")
                pos_label.pack(side=tk.LEFT)
                
                # 選択/移動ボタン
                if selected_player[0] is None:
                    select_btn = ttk.Button(player_frame, text="Select", width=12,
                                          command=lambda idx=i: select_player(idx))
                    select_btn.pack(side=tk.LEFT, padx=2)
                elif selected_player[0] == i:
                    cancel_btn = ttk.Button(player_frame, text="Cancel", width=12,
                                          command=lambda: cancel_selection())
                    cancel_btn.pack(side=tk.LEFT, padx=2)
                else:
                    move_btn = ttk.Button(player_frame, text=f"Move Here", width=12,
                                        command=lambda to_idx=i: move_player(selected_player[0], to_idx))
                    move_btn.pack(side=tk.LEFT, padx=2)
        
        def select_player(index):
            selected_player[0] = index
            refresh_order_display()
        
        def cancel_selection():
            selected_player[0] = None
            refresh_order_display()
        
        def move_player(from_idx, to_idx):
            success, message = lineup_manager.move_player_in_batting_order(from_idx, to_idx)
            if success:
                selected_player[0] = None
                refresh_order_display()
                refresh_callback()  # 親ダイアログも更新
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)
        
        # 初期表示
        refresh_order_display()
        
        # 完了ボタン
        ttk.Button(order_window, text="Done", command=order_window.destroy).pack(pady=10)

    def _execute_position_exchange(self, player1_idx, player2_idx):
        """2人の選手のポジションを交換"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
            
        team = self.game_state.fielding_team
        player1 = team.lineup[player1_idx]
        player2 = team.lineup[player2_idx]
        
        # 現在のポジションを取得
        pos1 = player1.current_position if hasattr(player1, 'current_position') and player1.current_position else player1.position
        pos2 = player2.current_position if hasattr(player2, 'current_position') and player2.current_position else player2.position
        
        # 相互の適性をチェック
        if not player1.can_play_position(pos2) or not player2.can_play_position(pos1):
            messagebox.showwarning("Warning", "Position exchange not possible due to player eligibility")
            return
        
        # 確認ダイアログ
        confirm = messagebox.askyesno("Position Exchange", 
            f"Exchange positions:\n"
            f"{player1.name}: {pos1} → {pos2}\n"
            f"{player2.name}: {pos2} → {pos1}\n\n"
            f"Do you want to proceed?")
        
        if not confirm:
            return
        
        # ポジションを交換
        if hasattr(player1, 'current_position'):
            player1.current_position = pos2
        else:
            player1.position = pos2
            
        if hasattr(player2, 'current_position'):
            player2.current_position = pos1
        else:
            player2.position = pos1
        
        # メインGUIの表示を更新
        if self.main_gui:
            self.main_gui._update_scoreboard(self.game_state)
        
        messagebox.showinfo("Position Exchange", 
            f"Position exchange completed:\n"
            f"{player1.name}: {pos1} → {pos2}\n"
            f"{player2.name}: {pos2} → {pos1}")
    
    def _start_defense_change_mode(self, substitution_manager):
        """新しい守備変更モードを開始"""
        # ゲーム終了チェック
        if self._is_game_over():
            messagebox.showinfo("Game Over", "The game has ended. No more strategies can be executed.")
            return
        
        try:
            # 守備変更モードインスタンスを作成
            defense_mode = DefenseChangeMode(self.root, substitution_manager, self.main_gui)
            defense_mode.start_defense_change_mode()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start defense change mode: {str(e)}")
