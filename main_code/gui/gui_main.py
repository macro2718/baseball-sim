"""
Main GUI class for the Baseball Simulation - Refactored
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import sys

# プロジェクト設定を使用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_code.config import setup_project_environment
setup_project_environment()

from .gui_constants import get_font_settings, get_ui_text
from .gui_field import FieldManager
from .gui_scoreboard import ScoreboardManager
from .gui_stats import StatsManager
from .gui_strategy import StrategyManager
from .gui_event_manager import Events
from main_code.config import UIConstants, GameResults

class BaseballGUI:
    def __init__(self, root, team_manager, event_manager, layout_manager):
        """
        BaseballGUIを初期化 - リファクタリング版
        
        Args:
            root: Tkinterのルートウィンドウ
            team_manager: チーム管理クラス
            event_manager: イベント管理クラス
            layout_manager: レイアウト管理クラス
        """
        self.root = root
        self.team_manager = team_manager
        self.event_manager = event_manager
        self.layout_manager = layout_manager
        
        # ゲーム状態
        self.game_state = None
        self.game_screen_created = False
        
        # UI設定
        self.text = get_ui_text()
        
        # 各種マネージャーの初期化
        self.field_manager = None
        self.scoreboard_manager = None
        self.strategy_manager = StrategyManager(self.root, self.text, self)
        self.stats_manager = StatsManager(self.root, self.text)
        
        # GUI要素の参照
        self.offense_players = []
        self.defense_players = []
        self.result_text = None
        self.normal_batting_button = None
        self.bunt_button = None
        self.offense_button = None
        self.defense_button = None
        self.stats_button = None
        
        # 画面要素の参照
        self.screen_elements = {}
        
        # イベントリスナーの登録
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """イベントリスナーを設定"""
        self.event_manager.bind(Events.TEAMS_CREATED, self.on_teams_created)
        self.event_manager.bind(Events.GAME_STARTED, self.on_game_started)
        self.event_manager.bind(Events.LINEUP_CHANGED, self.on_lineup_changed)
        self.event_manager.bind(Events.GAME_STATE_CHANGED, self.on_game_state_changed)
    
    def show_title_screen(self):
        """タイトル画面を表示"""
        self.layout_manager.create_title_screen(
            on_new_game=self._start_game,
            on_exit=self.root.quit
        )
    
    def _start_game(self):
        """ゲームを開始"""
        if not self.team_manager.has_teams():
            try:
                success, message = self.team_manager.create_teams()
                if not success:
                    messagebox.showerror("Error", message)
                    return
                print(message)
                self.event_manager.trigger(Events.TEAMS_CREATED)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create teams: {e}")
                return
        
        # スタメン設定をスキップして即座にゲームを開始
        self._proceed_to_game()
    
    def on_teams_created(self):
        """チーム作成完了時のイベントハンドラー"""
        print("Teams created successfully")
        # タイトル画面を更新（ラインナップボタンを表示）
        self.show_title_screen()
    
    def on_game_started(self, game_state):
        """ゲーム開始時のイベントハンドラー"""
        self.game_state = game_state
        self._create_game_screen()
        
        # 各マネージャーにゲーム状態を設定
        if self.field_manager:
            self.field_manager.set_game_state(game_state)
        if self.scoreboard_manager:
            self.scoreboard_manager.set_game_state(game_state)
        self.strategy_manager.set_game_state(game_state)
        self.stats_manager.set_game_state(game_state)
        
        # 初期化処理
        self._update_scoreboard(game_state)
        self._display_game_start_message()
        self._update_button_states()
    
    def on_lineup_changed(self, team_type):
        """ラインナップ変更時のイベントハンドラー"""
        print(f"Lineup changed for {team_type} team")
        # 必要に応じて画面を更新
    
    def on_game_state_changed(self, game_state):
        """ゲーム状態変更時のイベントハンドラー"""
        self.game_state = game_state
        if self.scoreboard_manager:
            self.scoreboard_manager.update_scoreboard(game_state)
        self._update_rosters(game_state)
        self._update_bunt_button_state()
    
    def _display_game_start_message(self):
        """ゲーム開始メッセージを表示"""
        if not self.game_state:
            return
            
        home_team, away_team = self.team_manager.get_teams()
        initial_message = self.text["game_start"].format(away_team.name, home_team.name)
        side_message = f"\n=== {'TOP' if self.game_state.is_top_inning else 'BOTTOM'} of the {self.game_state.inning} ==="
        
        if self.result_text:
            self._add_result_text(initial_message)
            self._add_result_text(side_message, "yellow")
    
    def _create_game_screen(self):
        """ゲーム画面を作成 - リファクタリング版"""
        if self.game_screen_created:
            return
            
        # レイアウトマネージャーを使用して画面構築
        self.screen_elements = self.layout_manager.create_game_screen_layout()
        
        # フィールドマネージャーの初期化
        self.field_manager = FieldManager(self.screen_elements['field_canvas'], self.text)
        self.field_manager.draw_field()
        
        # 選手名簿の初期化
        self._setup_roster_displays()
        
        # スコアボードマネージャーの初期化
        self.scoreboard_manager = ScoreboardManager(self.screen_elements['scoreboard_frame'], self.text)
        self.scoreboard_manager.create_scoreboard()
        self.scoreboard_manager.create_situation_panel(self.screen_elements['situation_frame'])
        
        # コントロールボタンの作成
        self._create_control_buttons()
        
        # 結果表示エリアの作成
        self._create_result_display()
        
        self.game_screen_created = True
    
    def _setup_roster_displays(self):
        """選手名簿表示を設定"""
        # 攻撃側選手リスト
        self.offense_players = []
        for i in range(UIConstants.ROSTER_DISPLAY_COUNT):
            player_label = ttk.Label(self.screen_elements['offense_roster_frame'], text="", anchor=tk.W)
            player_label.pack(fill=tk.X, padx=5, pady=1)
            self.offense_players.append(player_label)
        
        # 守備側選手リスト
        self.defense_players = []
        for i in range(UIConstants.ROSTER_DISPLAY_COUNT):
            player_label = ttk.Label(self.screen_elements['defense_roster_frame'], text="", anchor=tk.W)
            player_label.pack(fill=tk.X, padx=5, pady=1)
            self.defense_players.append(player_label)
    
    def _create_control_buttons(self):
        """コントロールボタンを作成"""
        control_frame = self.screen_elements['control_frame']
        play_frame = self.screen_elements['play_frame']
        
        # 采配ボタン
        self.offense_button = ttk.Button(
            control_frame, 
            text=self.text["offense_strategy"], 
            command=self.strategy_manager.show_offense_menu,
            state=tk.DISABLED
        )
        self.offense_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.defense_button = ttk.Button(
            control_frame, 
            text=self.text["defense_strategy"], 
            command=self.strategy_manager.show_defense_menu,
            state=tk.DISABLED
        )
        self.defense_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 選手成績表示ボタン
        self.stats_button = ttk.Button(
            control_frame, 
            text=self.text["stats"], 
            command=self.stats_manager.show_player_stats,
            state=tk.DISABLED
        )
        self.stats_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 通常打撃ボタン
        self.normal_batting_button = ttk.Button(
            play_frame, 
            text="Normal Batting (Game not started)", 
            command=self._execute_normal_batting, 
            state=tk.DISABLED
        )
        self.normal_batting_button.pack(fill=tk.X, padx=5, pady=2)
        
        # バントボタン
        self.bunt_button = ttk.Button(
            play_frame, 
            text="Bunt (Game not started)", 
            command=self._execute_bunt_from_dialog, 
            state=tk.DISABLED
        )
        self.bunt_button.pack(fill=tk.X, padx=5, pady=2)
    
    def _create_result_display(self):
        """結果表示エリアを作成"""
        result_frame = self.screen_elements['result_frame']
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=10
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
    
    def _update_rosters(self, game_state):
        """選手名簿の表示を更新"""
        # 攻撃側の選手名簿を更新
        batting_team = game_state.batting_team
        if 'offense_roster_frame' in self.screen_elements:
            self.screen_elements['offense_roster_frame'].config(text=f"Offense: {batting_team.name}")
        
        for i, label in enumerate(self.offense_players):
            if i < len(batting_team.lineup):
                player = batting_team.lineup[i]
                # 現在の守備位置と守備可能位置を表示
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                current_pos = player.current_position if hasattr(player, 'current_position') and player.current_position else primary_pos
                eligible_positions = player.get_display_eligible_positions() if hasattr(player, 'get_display_eligible_positions') else [primary_pos]
                
                # 守備可能位置をカンマ区切りで表示し、実際の守備位置を表示
                eligible_str = ", ".join(eligible_positions)
                position_display = f"Eligible: {eligible_str}"
                
                # 現在の打者には星印をつける
                if i == batting_team.current_batter_index:
                    label.config(text=f"★ {i+1}. {current_pos} | {player.name} [{position_display}]")
                    label.config(foreground="blue")
                else:
                    label.config(text=f"{i+1}. {current_pos} | {player.name} [{position_display}]")
                    label.config(foreground="black")
            else:
                label.config(text="")
        
        # 守備側の選手名簿を更新
        fielding_team = game_state.fielding_team
        if 'defense_roster_frame' in self.screen_elements:
            self.screen_elements['defense_roster_frame'].config(text=f"Defense: {fielding_team.name}")
        
        for i, label in enumerate(self.defense_players):
            if i < len(fielding_team.lineup):
                player = fielding_team.lineup[i]
                # 現在の守備位置と守備可能位置を表示
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                current_pos = player.current_position if hasattr(player, 'current_position') and player.current_position else primary_pos
                eligible_positions = player.get_display_eligible_positions() if hasattr(player, 'get_display_eligible_positions') else [primary_pos]
                
                # 守備可能位置をカンマ区切りで表示し、実際の守備位置を表示
                eligible_str = ", ".join(eligible_positions)
                position_display = f"Eligible: {eligible_str}"
                
                # 現在の投手には星印をつける
                if player == fielding_team.current_pitcher:
                    label.config(text=f"★ {i+1}. {current_pos} | {player.name} [{position_display}]")
                    label.config(foreground="red")
                else:
                    label.config(text=f"{i+1}. {current_pos} | {player.name} [{position_display}]")
                    label.config(foreground="black")
            else:
                label.config(text="")
    
    def _update_scoreboard(self, game_state):
        """スコアボードの情報を更新"""
        # ゲーム画面が作成されていない場合は何もしない
        if not self.game_screen_created:
            return
        
        # 選手名簿を更新
        self._update_rosters(game_state)
        
        # スコアボードを更新
        self.scoreboard_manager.update_scoreboard(game_state)
        
        # アウトカウントを表示
        self.field_manager.update_outs_display(game_state.outs)
        
        # フィールド表示更新
        self.field_manager.update_field(game_state.bases)
        
        # ゲーム終了チェック
        if self._is_game_over():
            # ゲーム終了時は戦略ボタンを無効化
            if hasattr(self, 'offense_button'):
                self.offense_button.config(state=tk.DISABLED, text="Game Over")
            if hasattr(self, 'defense_button'):
                self.defense_button.config(state=tk.DISABLED, text="Game Over")
            if hasattr(self, 'normal_batting_button'):
                self.normal_batting_button.config(state=tk.DISABLED, text="Game Over")
            if hasattr(self, 'bunt_button'):
                self.bunt_button.config(state=tk.DISABLED, text="Game Over")
        else:
            # ゲーム継続中は通常の状態更新
            # バントボタンの状態を更新
            self._update_bunt_button_state()
            
            # 通常打撃ボタンの状態を更新
            if hasattr(self, 'normal_batting_button') and self.game_state:
                self.normal_batting_button.config(state=tk.NORMAL, text="Normal Batting")
    
    def _is_game_over(self):
        """ゲームが終了しているかチェック"""
        if not self.game_state:
            return False
        
        return self.game_state.game_ended
    
    def _add_result_text(self, text, color=None):
        """結果テキストエリアにテキストを追加"""
        self.result_text.config(state=tk.NORMAL)
        
        # 色タグを設定
        if color == "blue":
            self.result_text.tag_config("blue", foreground="blue")
            self.result_text.insert(tk.END, text, "blue")
        elif color == "red":
            self.result_text.tag_config("red", foreground="red")
            self.result_text.insert(tk.END, text, "red")
        elif color == "yellow":
            self.result_text.tag_config("yellow", foreground="orange", font=("Helvetica", 10, "bold"))
            self.result_text.insert(tk.END, text, "yellow")
        else:
            self.result_text.insert(tk.END, text)
            
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)
    
    def _update_bunt_button_state(self):
        """バントボタンの有効/無効状態を更新"""
        if not self.game_state or not hasattr(self, 'bunt_button'):
            # ゲーム状態が設定されていない場合はボタンを無効化
            if hasattr(self, 'bunt_button'):
                self.bunt_button.config(state=tk.DISABLED, text="Bunt (Game not started)")
            return
        
        # バント可能かチェック
        can_bunt = self.game_state.can_bunt()
        
        if can_bunt:
            self.bunt_button.config(state=tk.NORMAL)
            # バント成功確率を取得してツールチップとして表示
            try:
                current_batter = self.game_state.batting_team.lineup[self.game_state.batting_team.current_batter_index]
                current_pitcher = self.game_state.fielding_team.current_pitcher
                from main_code.core.game_utils import BuntCalculator
                bunt_success_rate = BuntCalculator.calculate_bunt_success_probability(current_batter, current_pitcher)
                self.bunt_button.config(text=f"Bunt ({bunt_success_rate:.1%})")
            except:
                self.bunt_button.config(text="Bunt")
        else:
            self.bunt_button.config(state=tk.DISABLED)
            # 無効化の理由を表示
            if not any(self.game_state.bases):
                self.bunt_button.config(text="Bunt (No Runners)")
            elif self.game_state.outs >= 2:
                self.bunt_button.config(text="Bunt (2 Outs)")
            else:
                self.bunt_button.config(text="Bunt (Unavailable)")
    
    def _get_current_situation_text(self):
        """現在の状況を文字列で取得"""
        runners_text = []
        if self.game_state.bases[0]:
            runners_text.append("1st")
        if self.game_state.bases[1]:
            runners_text.append("2nd")
        if self.game_state.bases[2]:
            runners_text.append("3rd")
        
        if not runners_text:
            situation = "No Runners"
        else:
            situation = f"Runners: {', '.join(runners_text)}"
        
        outs_text = f"{self.game_state.outs} Outs"
        inning_text = f"{self.game_state.inning}{'T' if self.game_state.is_top_inning else 'B'}"
        
        return f"{inning_text} {outs_text} {situation}"
    
    def _execute_normal_batting(self):
        """通常の打撃を実行"""
        if not self.game_state:
            self._add_result_text("\nGame not started. Please start a new game.\n", "red")
            return
        
        # ゲーム終了チェック
        if self._is_game_over():
            self._add_result_text("\nThe game has ended. No more batting actions allowed.\n", "red")
            return
            
        # 守備位置エラーチェック - ゲームアクションが許可されているかの確認
        allowed, error_msg = self.game_state.is_game_action_allowed()
        if not allowed:
            self._add_result_text(f"\n❌ {error_msg}\n", "red")
            self._add_result_text("Please fix the defensive position issues before continuing.\n", "yellow")
            return
        
        # 攻守交代が起きるかチェック（イニングの変わり目）
        prev_inning = self.game_state.inning
        prev_is_top = self.game_state.is_top_inning
        
        batter = self.game_state.batting_team.lineup[self.game_state.batting_team.current_batter_index]
        pitcher = self.game_state.fielding_team.current_pitcher
        
        result = self.game_state.calculate_result(batter, pitcher)
        message = self.game_state.apply_result(result, batter)
        
        # 結果に応じて色分け（定数を使用）
        if result in GameResults.POSITIVE_RESULTS:
            # 良い結果は青色
            self._add_result_text(f"\n{batter.name} vs {pitcher.name}\n")
            self._add_result_text(message, "blue")
        else:
            # 悪い結果は赤色
            self._add_result_text(f"\n{batter.name} vs {pitcher.name}\n")
            self._add_result_text(message, "red")
        
        # アニメーション効果
        self.field_manager.animate_play_result(result, self.root)
        
        # 投手のスタミナを減少
        pitcher.decrease_stamina()
        
        # 攻守交代を検出
        inning_changed = prev_inning != self.game_state.inning or prev_is_top != self.game_state.is_top_inning
        
        # 攻守交代が発生しなかった場合のみ次の打者へ
        if not inning_changed:
            self.game_state.batting_team.next_batter()
        
        # 攻守交代を検出してメッセージを表示
        if inning_changed:
            side_message = f"\n=== {'TOP' if self.game_state.is_top_inning else 'BOTTOM'} of the {self.game_state.inning} ==="
            self._add_result_text(side_message, "yellow")
            # 攻守交代のアニメーション表示
            self.field_manager.animate_change(self.root)
        
        # ゲーム終了判定
        if self.game_state.game_ended:
            self._game_over()
        
        # スコアボードとフィールド表示の更新
        self._update_scoreboard(self.game_state)
    
    def _execute_bunt_from_dialog(self):
        """ダイアログからのバント実行"""
        if not self.game_state:
            self._add_result_text("\nGame not started. Please start a new game.\n", "red")
            return
        
        # ゲーム終了チェック
        if self._is_game_over():
            self._add_result_text("\nThe game has ended. No more batting actions allowed.\n", "red")
            return
            
        # 守備位置エラーチェック - ゲームアクションが許可されているかの確認
        allowed, error_msg = self.game_state.is_game_action_allowed()
        if not allowed:
            self._add_result_text(f"\n❌ {error_msg}\n", "red")
            self._add_result_text("Please fix the defensive position issues before continuing.\n", "yellow")
            return
        
        # バント可能性をチェック
        if not self.game_state.can_bunt():
            # バントが無効な場合はメッセージを表示して戻る
            if not any(self.game_state.bases):
                self._add_result_text("\nCannot bunt (No runners on base)\n", "red")
            elif self.game_state.outs >= 2:
                self._add_result_text("\nCannot bunt (2 outs)\n", "red")
            return
        
        # 攻守交代が起きるかチェック（イニングの変わり目）
        prev_inning = self.game_state.inning
        prev_is_top = self.game_state.is_top_inning
        
        batter = self.game_state.batting_team.lineup[self.game_state.batting_team.current_batter_index]
        pitcher = self.game_state.fielding_team.current_pitcher
        
        # バント結果を取得
        result_message = self.game_state.execute_bunt(batter, pitcher)
        
        # バント失敗メッセージの場合は早期リターン
        if "Cannot bunt" in result_message or "バントはできません" in result_message:
            self._add_result_text(f"\n{result_message}\n", "red")
            return
        
        # 結果を表示
        self._add_result_text(f"\n{batter.name} Bunt vs {pitcher.name}\n")
        self._add_result_text(result_message, "green")
        
        # バントのアニメーション（通常の打撃と区別）
        self.field_manager.animate_play_result("bunt", self.root)
        
        # 投手のスタミナを減少
        pitcher.decrease_stamina()
        
        # 次の打者へ
        self.game_state.batting_team.next_batter()
        
        # 攻守交代を検出してメッセージを表示
        if prev_inning != self.game_state.inning or prev_is_top != self.game_state.is_top_inning:
            side_message = f"\n=== {'TOP' if self.game_state.is_top_inning else 'BOTTOM'} of the {self.game_state.inning} ==="
            self._add_result_text(side_message, "yellow")
            # 攻守交代のアニメーション表示
            self.field_manager.animate_change(self.root)
        
        # ゲーム終了判定
        if self.game_state.inning >= 9 and not self.game_state.is_top_inning:
            if self.game_state.home_score > self.game_state.away_score:
                self._game_over()
        
        if self.game_state.inning > 12:
            self._game_over()
        
        # スコアボードとフィールド表示の更新
        self._update_scoreboard(self.game_state)
    
    def _game_over(self):
        """ゲーム終了処理"""
        home_team, away_team = self.team_manager.get_teams()
        
        result_text = f"\n{self.text['game_over']}\n"
        result_text += self.text["final_score"].format(
            away_team.name, self.game_state.away_score, 
            self.game_state.home_score, home_team.name
        )
        result_text += "\n"
        
        if self.game_state.home_score > self.game_state.away_score:
            result_text += self.text["victory"].format(home_team.name)
        elif self.game_state.away_score > self.game_state.home_score:
            result_text += self.text["victory"].format(away_team.name)
        else:
            result_text += self.text["tie"]
        
        self._add_result_text(result_text)
        
        # 試合終了ダイアログ
        messagebox.showinfo(self.text["game_over"], result_text)
        
        # 再スタートボタンを表示
        if 'main_frame' in self.screen_elements:
            restart_button = ttk.Button(self.screen_elements['main_frame'], 
                                      text=self.text["new_game"], 
                                      command=self.show_title_screen)
            restart_button.pack(pady=10)
        
        # すべてのゲーム操作ボタンを無効化
        if hasattr(self, 'normal_batting_button'):
            self.normal_batting_button.config(state=tk.DISABLED, text="Game Over")
        if hasattr(self, 'bunt_button'):
            self.bunt_button.config(state=tk.DISABLED, text="Game Over")
        if hasattr(self, 'offense_button'):
            self.offense_button.config(state=tk.DISABLED, text="Game Over")
        if hasattr(self, 'defense_button'):
            self.defense_button.config(state=tk.DISABLED, text="Game Over")
        
        # 選手成績ボタンは有効のまま維持（ゲーム終了後も確認可能）
        # stats_buttonは無効化しない
    
    def set_teams_and_start_game(self, home_team, away_team, game_state):
        """チーム情報とゲーム状態を設定してゲームを開始"""
        # チームマネージャーに設定
        self.team_manager.set_teams(home_team, away_team)
        self.game_state = game_state
        
        # 各マネージャーにゲーム状態を設定
        if self.field_manager:
            self.field_manager.set_game_state(game_state)
        if self.scoreboard_manager:
            self.scoreboard_manager.set_game_state(game_state)
        self.strategy_manager.set_game_state(game_state)
        self.stats_manager.set_game_state(game_state)
        
        # ゲーム画面を作成
        self._create_game_screen()
        
        # スコアボードと選手名簿を更新
        self._update_scoreboard(game_state)
        
        # 初期メッセージを表示
        initial_message = self.text["game_start"].format(away_team.name, home_team.name)
        side_message = f"\n=== {'TOP' if game_state.is_top_inning else 'BOTTOM'} of the {game_state.inning} ==="
        
        if hasattr(self, 'result_text'):
            self._add_result_text(initial_message)
            self._add_result_text(side_message, "yellow")
    
    def set_teams(self, home_team, away_team):
        """チーム情報のみを設定"""
        # チームマネージャーに設定
        self.team_manager.set_teams(home_team, away_team)
        # チーム設定後にタイトル画面を再描画
        self.show_title_screen()
    
    def set_game_state(self, game_state):
        """ゲーム状態を設定"""
        self.game_state = game_state
        
        # 各マネージャーにゲーム状態を設定
        if self.field_manager:
            self.field_manager.set_game_state(game_state)
        if self.scoreboard_manager:
            self.scoreboard_manager.set_game_state(game_state)
        self.strategy_manager.set_game_state(game_state)
        self.stats_manager.set_game_state(game_state)
        
        # ゲーム画面が作成されていない場合は作成
        if not self.game_screen_created:
            self._create_game_screen()
        
        # スコアボードを更新
        if self.scoreboard_manager:
            self.scoreboard_manager.update_scoreboard(game_state)
        
        # バントボタンの状態を更新
        self._update_bunt_button_state()
    

    

    
    def _proceed_to_game(self):
        """ゲーム画面に進む"""
        # ゲーム状態を作成
        home_team, away_team = self.team_manager.get_teams()
        if not self.game_state and home_team and away_team:
            from main_code.core.game import GameState
            self.game_state = GameState(home_team, away_team)
            
            # 各マネージャーにゲーム状態を設定
            self.strategy_manager.set_game_state(self.game_state)
            self.stats_manager.set_game_state(self.game_state)
        
        # ゲーム画面を作成
        self._create_game_screen()
        
        # ゲーム状態が既にセットされている場合はスコアボードを更新
        if self.game_state:
            # 各マネージャーにゲーム状態を設定
            if self.field_manager:
                self.field_manager.set_game_state(self.game_state)
            if self.scoreboard_manager:
                self.scoreboard_manager.set_game_state(self.game_state)
            
            # スコアボードと選手名簿を更新
            self._update_scoreboard(self.game_state)
            
            # 初期メッセージを表示
            initial_message = self.text["game_start"].format(away_team.name, home_team.name)
            side_message = f"\n=== {'TOP' if self.game_state.is_top_inning else 'BOTTOM'} of the {self.game_state.inning} ==="
            
            if hasattr(self, 'result_text'):
                self._add_result_text(initial_message)
                self._add_result_text(side_message, "yellow")
            
            # ボタンの状態を更新
            self._update_button_states()
            
            print("Game started successfully with configured lineups!")

    def _create_teams(self):
        """チームを作成 - 新しいアーキテクチャ用"""
        success, message = self.team_manager.create_teams()
        if success:
            print(message)
            self.event_manager.trigger(Events.TEAMS_CREATED)
        else:
            from tkinter import messagebox
            messagebox.showerror("Error", message)
            raise Exception(message)
    

    

    

    

    

    

    
    def _update_button_states(self):
        """ゲーム開始後のボタン状態を更新"""
        if not self.game_state:
            return
        
        # 通常打撃ボタンの状態を更新
        if hasattr(self, 'normal_batting_button'):
            self.normal_batting_button.config(state=tk.NORMAL, text="Normal Batting")
        
        # バントボタンの状態を更新
        self._update_bunt_button_state()
        
        # 戦略ボタンの状態を更新
        if hasattr(self, 'offense_button'):
            self.offense_button.config(state=tk.NORMAL)
        if hasattr(self, 'defense_button'):
            self.defense_button.config(state=tk.NORMAL)
        if hasattr(self, 'stats_button'):
            self.stats_button.config(state=tk.NORMAL)
    
