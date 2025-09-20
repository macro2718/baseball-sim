"""Main GUI class for the Baseball Simulation."""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from baseball_sim.config import GameResults, UIConstants, setup_project_environment
from baseball_sim.infrastructure.logging_utils import logger

from baseball_sim.gameplay.game import GameState

from .gui_constants import get_ui_text
from .gui_event_manager import Events
from .gui_field import FieldManager
from .gui_scoreboard import ScoreboardManager
from .gui_stats import StatsManager
from .gui_strategy import StrategyManager
from .gui_colors import get_position_color

setup_project_environment()

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
        self.title_screen_elements = {}
        self.toolbar_controls = {}
        self.status_label_widget = None
        self.action_hint_label = None

        # 補助状態
        self.log_auto_scroll = tk.BooleanVar(value=True)
        self.keyboard_bindings_active = False
        self._pending_log_reset = False

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
        self._unbind_keyboard_shortcuts()
        self.title_screen_elements = self.layout_manager.create_title_screen(
            on_new_game=self._start_game,
            on_exit=self.root.quit
        )

        status_rows = self.title_screen_elements.get("status_rows", {})
        if "home" in status_rows:
            status_rows["home"]["detail_button"].config(
                command=lambda: self._show_team_preview('home')
            )
        if "away" in status_rows:
            status_rows["away"]["detail_button"].config(
                command=lambda: self._show_team_preview('away')
            )

        self._update_title_screen_status()

    def _update_title_screen_status(self):
        """タイトル画面のチームステータスを更新"""
        if not self.title_screen_elements:
            return

        rows = self.title_screen_elements.get("status_rows", {})
        start_button = self.title_screen_elements.get("start_button")
        hint_label = self.title_screen_elements.get("hint_label")

        has_teams = self.team_manager.has_teams()
        invalid_messages = []

        for team_key in ("away", "home"):
            widgets = rows.get(team_key)
            if not widgets:
                continue

            team = self.team_manager.get_team_by_type(team_key)
            label_suffix = self.text[team_key]

            if team:
                widgets["name_label"].config(text=f"{team.name} ({label_suffix})")
                status = self.team_manager.get_team_status(team_key)
                widgets["state_label"].config(text=status['message'], foreground=status['color'])
                widgets["detail_button"].config(state=tk.NORMAL)

                if not status['valid'] and status['errors']:
                    invalid_messages.append(f"{team.name}: {len(status['errors'])}")
            else:
                widgets["name_label"].config(text=f"- ({label_suffix})")
                widgets["state_label"].config(text=self.text["team_status_unknown"], foreground="grey")
                widgets["detail_button"].config(state=tk.DISABLED)

        if start_button:
            # 無効なラインナップがある場合は注意喚起
            if has_teams and invalid_messages:
                start_button.config(text=f"{self.text['new_game']} ⚠")
            else:
                start_button.config(text=self.text['new_game'])

        if hint_label:
            if not has_teams:
                hint_label.config(text=self.text["title_hint"])
            elif invalid_messages:
                hint_label.config(text=self.text["title_hint_with_errors"].format(", ".join(invalid_messages)))
            else:
                hint_label.config(text=self.text["title_hint_ready"])

    def _load_teams(self, force_reload=False, notify=True):
        """Ensure teams are loaded, optionally forcing a fresh load."""
        if force_reload and hasattr(self.team_manager, 'reset_teams'):
            self.team_manager.reset_teams()

        if not force_reload and self.team_manager.has_teams():
            return True

        try:
            success, message = self.team_manager.create_teams()
        except Exception as exc:
            logger.error(f"Failed to create teams: {exc}")
            messagebox.showerror("Error", f"Failed to create teams: {exc}")
            return False

        if not success:
            messagebox.showerror("Error", message)
            return False

        logger.info(message)
        if notify:
            self.event_manager.trigger(Events.TEAMS_CREATED)
        return True

    def _start_new_game_session(self, game_state, reset_log=False):
        """Set up UI and managers for a fresh game session."""
        self.game_state = game_state

        if reset_log:
            self._clear_result_display(add_marker=False)

        self._create_game_screen()
        self.strategy_manager.set_game_state(game_state)
        self.stats_manager.set_game_state(game_state)
        if self.field_manager:
            self.field_manager.set_game_state(game_state)
        if self.scoreboard_manager:
            self.scoreboard_manager.set_game_state(game_state)
        self._update_scoreboard(game_state)
        self._display_game_start_message()
        self._update_button_states()

        self._pending_log_reset = False

    def _reset_game_references(self, clear_teams=False):
        """Reset UI references so title screen can rebuild cleanly."""
        self._unbind_keyboard_shortcuts()

        self.game_screen_created = False
        self.field_manager = None
        self.scoreboard_manager = None
        self.result_text = None
        self.normal_batting_button = None
        self.bunt_button = None
        self.offense_button = None
        self.defense_button = None
        self.stats_button = None
        self.toolbar_controls = {}
        self.status_label_widget = None
        self.action_hint_label = None
        self.screen_elements = {}
        self.game_state = None
        self._pending_log_reset = False

        if clear_teams and hasattr(self.team_manager, 'reset_teams'):
            self.team_manager.reset_teams()
    
    def _start_game(self):
        """ゲームを開始"""
        if not self._load_teams():
            return

        self._proceed_to_game()
    
    def on_teams_created(self):
        """チーム作成完了時のイベントハンドラー"""
        logger.info("Teams created successfully")
        # タイトル画面を更新（ラインナップボタンを表示）
        self.show_title_screen()
    
    def on_game_started(self, game_state):
        """ゲーム開始時のイベントハンドラー"""
        self.game_state = game_state
        reset_log = self._pending_log_reset
        self._start_new_game_session(game_state, reset_log=reset_log)
    
    def on_lineup_changed(self, team_type):
        """ラインナップ変更時のイベントハンドラー"""
        logger.info(f"Lineup changed for {team_type} team")
        # 必要に応じて画面を更新
        self._update_title_screen_status()
    
    def on_game_state_changed(self, game_state):
        """ゲーム状態変更時のイベントハンドラー"""
        self.game_state = game_state
        if self.scoreboard_manager:
            self.scoreboard_manager.update_scoreboard(game_state)
        self._update_rosters(game_state)
        self._update_bunt_button_state()
        self._update_status_bar()
        self._update_toolbar_state()
    
    def _display_game_start_message(self):
        """ゲーム開始メッセージを表示"""
        if not self.game_state:
            return
            
        home_team, away_team = self.team_manager.get_teams()
        initial_message = self.text["game_start"].format(away_team.name, home_team.name)
        side_message = f"\n=== {'TOP' if self.game_state.is_top_inning else 'BOTTOM'} of the {self.game_state.inning} ==="
        
        if self.result_text:
            self._add_result_text(initial_message)
            self._add_result_text(side_message, "orange")
    
    def _create_game_screen(self):
        """ゲーム画面を作成 - リファクタリング版"""
        if self.game_screen_created:
            return
            
        # レイアウトマネージャーを使用して画面構築
        self.screen_elements = self.layout_manager.create_game_screen_layout()
        self.status_label_widget = self.screen_elements.get('status_label')
        self.action_hint_label = self.screen_elements.get('action_label')
        self._build_toolbar(self.screen_elements.get('toolbar_frame'))

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
        self._update_status_bar()
        self._bind_keyboard_shortcuts()

    def _build_toolbar(self, toolbar_frame):
        """ゲーム画面上部のツールバーを構築"""
        if not toolbar_frame:
            return

        for widget in toolbar_frame.winfo_children():
            widget.destroy()

        left_frame = ttk.Frame(toolbar_frame)
        left_frame.pack(side=tk.LEFT, padx=5)

        right_frame = ttk.Frame(toolbar_frame)
        right_frame.pack(side=tk.RIGHT, padx=5)

        restart_btn = ttk.Button(left_frame, text=self.text["toolbar_new_game"], command=self._restart_game)
        restart_btn.pack(side=tk.LEFT, padx=3)

        return_btn = ttk.Button(left_frame, text=self.text["toolbar_return_title"], command=self._return_to_title)
        return_btn.pack(side=tk.LEFT, padx=3)

        clear_btn = ttk.Button(left_frame, text=self.text["toolbar_clear_log"], command=self._clear_result_display)
        clear_btn.pack(side=tk.LEFT, padx=3)

        copy_btn = ttk.Button(left_frame, text=self.text["toolbar_copy_log"], command=self._copy_result_display)
        copy_btn.pack(side=tk.LEFT, padx=3)

        auto_scroll_btn = ttk.Checkbutton(right_frame, text=self.text["toolbar_autoscroll"], variable=self.log_auto_scroll)
        auto_scroll_btn.pack(side=tk.RIGHT, padx=3)

        self.toolbar_controls = {
            'restart': restart_btn,
            'return': return_btn,
            'clear': clear_btn,
            'copy': copy_btn,
            'auto_scroll': auto_scroll_btn,
        }

        self._update_toolbar_state()

    def _update_toolbar_state(self):
        """ツールバーのボタン状態を更新"""
        if not self.toolbar_controls:
            return

        game_ready = self.game_state is not None

        if 'restart' in self.toolbar_controls:
            self.toolbar_controls['restart'].config(state=tk.NORMAL if game_ready else tk.DISABLED)

        # 返すボタンは常に有効
        if 'clear' in self.toolbar_controls:
            self.toolbar_controls['clear'].config(state=tk.NORMAL if self.result_text else tk.DISABLED)

        if 'copy' in self.toolbar_controls:
            has_text = bool(self.result_text and self.result_text.get("1.0", tk.END).strip())
            self.toolbar_controls['copy'].config(state=tk.NORMAL if has_text else tk.DISABLED)

        if 'auto_scroll' in self.toolbar_controls:
            self.toolbar_controls['auto_scroll'].config(state=tk.NORMAL if self.result_text else tk.DISABLED)

    def _update_status_bar(self, custom_text=None):
        """ステータスバーの表示を更新"""
        if not self.status_label_widget or not self.action_hint_label:
            return

        if custom_text is not None:
            status_text = custom_text
        elif self.game_state:
            status_text = self._get_current_situation_text()
        else:
            status_text = self.text["status_waiting"]

        self.status_label_widget.config(text=status_text)

        if self.game_state and not self._is_game_over():
            action_text = self.text["status_default_hint"]
        elif self.game_state and self._is_game_over():
            action_text = self.text["status_restart_hint"]
        else:
            action_text = ""

        self.action_hint_label.config(text=action_text)

    def _clear_result_display(self, add_marker=True):
        """結果ログをクリア"""
        if not self.result_text:
            return

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        if add_marker:
            self.result_text.insert(tk.END, self.text["log_cleared"])
        self.result_text.config(state=tk.DISABLED)

        if self.log_auto_scroll.get():
            self.result_text.see(tk.END)

        self._update_toolbar_state()

    def _copy_result_display(self):
        """結果ログをクリップボードにコピー"""
        if not self.result_text:
            return

        content = self.result_text.get("1.0", tk.END).strip()
        if not content:
            return

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
        except Exception as exc:
            logger.debug(f"Failed to copy log to clipboard: {exc}")

        self._update_toolbar_state()

    def _open_stats_window(self):
        """成績表示ダイアログを開く"""
        if not self.game_state:
            messagebox.showinfo("Info", "Game has not started yet.")
            return
        self.stats_manager.show_player_stats()

    def _return_to_title(self):
        """タイトル画面に戻る"""
        self._reset_game_references(clear_teams=True)
        self.show_title_screen()
        self._update_status_bar()

    def _restart_game(self):
        """試合をリスタート"""
        if not self._load_teams(force_reload=True, notify=False):
            return

        home_team, away_team = self.team_manager.get_teams()
        if not home_team or not away_team:
            messagebox.showinfo("Info", "Teams are not ready yet.")
            return

        try:
            self.game_state = GameState(home_team, away_team)
        except Exception as exc:
            logger.error(f"Failed to restart game: {exc}")
            messagebox.showerror("Error", f"Failed to restart game: {exc}")
            return
        self._pending_log_reset = True
        self.event_manager.trigger(Events.GAME_STARTED, self.game_state)

    def _bind_keyboard_shortcuts(self):
        """キーボードショートカットをバインド"""
        if self.keyboard_bindings_active:
            return

        self.root.bind('<space>', self._handle_space_shortcut)
        self.root.bind('<Key-b>', self._handle_bunt_shortcut)
        self.root.bind('<Key-B>', self._handle_bunt_shortcut)
        self.keyboard_bindings_active = True

    def _unbind_keyboard_shortcuts(self):
        """キーボードショートカットを解除"""
        if not self.keyboard_bindings_active:
            return

        self.root.unbind('<space>')
        self.root.unbind('<Key-b>')
        self.root.unbind('<Key-B>')
        self.keyboard_bindings_active = False

    def _handle_space_shortcut(self, event):
        """スペースキーショートカット"""
        if self.game_state and not self._is_game_over():
            self._execute_normal_batting()
            return "break"
        return None

    def _handle_bunt_shortcut(self, event):
        """Bキーショートカット"""
        if self.game_state and not self._is_game_over():
            self._execute_bunt_from_dialog()
            return "break"
        return None
    
    def _setup_roster_displays(self):
        """選手名簿表示を設定"""
        # 攻撃側選手リスト（部分着色対応のためTextを使用）
        self.offense_players = []
        for i in range(UIConstants.ROSTER_DISPLAY_COUNT):
            text_widget = tk.Text(self.screen_elements['offense_roster_frame'], height=1, wrap=tk.NONE)
            text_widget.configure(state=tk.DISABLED)
            text_widget.pack(fill=tk.X, padx=5, pady=1)
            self.offense_players.append(text_widget)
        
        # 守備側選手リスト（部分着色対応のためTextを使用）
        self.defense_players = []
        for i in range(UIConstants.ROSTER_DISPLAY_COUNT):
            text_widget = tk.Text(self.screen_elements['defense_roster_frame'], height=1, wrap=tk.NONE)
            text_widget.configure(state=tk.DISABLED)
            text_widget.pack(fill=tk.X, padx=5, pady=1)
            self.defense_players.append(text_widget)
    
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
            command=self._open_stats_window,
            state=tk.DISABLED
        )
        self.stats_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 通常打撃ボタン
        self.normal_batting_button = ttk.Button(
            play_frame,
            text=self.text["normal_batting"],
            command=self._execute_normal_batting,
            state=tk.DISABLED
        )
        self.normal_batting_button.pack(fill=tk.X, padx=5, pady=2)
        
        # バントボタン
        self.bunt_button = ttk.Button(
            play_frame,
            text=self.text["bunt_action"],
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
        self._update_toolbar_state()
    
    def _update_rosters(self, game_state):
        """選手名簿の表示を更新"""
        # 攻撃側の選手名簿を更新
        batting_team = game_state.batting_team
        if 'offense_roster_frame' in self.screen_elements:
            self.screen_elements['offense_roster_frame'].config(text=f"Offense: {batting_team.name}")
        
        for i, text_widget in enumerate(self.offense_players):
            if i < len(batting_team.lineup):
                player = batting_team.lineup[i]
                # 現在の守備位置と守備可能位置を表示
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                current_pos = player.current_position if hasattr(player, 'current_position') and player.current_position else primary_pos
                eligible_positions = player.get_display_eligible_positions() if hasattr(player, 'get_display_eligible_positions') else [primary_pos]
                
                # ローラインを部分着色で描画
                is_current_batter = (i == batting_team.current_batter_index)
                self._render_roster_line(
                    text_widget,
                    line_index=i + 1,
                    star=is_current_batter,
                    current_pos=current_pos,
                    player=player,
                    eligible_positions=eligible_positions,
                )
            else:
                self._clear_roster_line(text_widget)
        
        # 守備側の選手名簿を更新
        fielding_team = game_state.fielding_team
        if 'defense_roster_frame' in self.screen_elements:
            self.screen_elements['defense_roster_frame'].config(text=f"Defense: {fielding_team.name}")
        
        for i, text_widget in enumerate(self.defense_players):
            if i < len(fielding_team.lineup):
                player = fielding_team.lineup[i]
                # 現在の守備位置と守備可能位置を表示
                primary_pos = player.eligible_positions[0] if player.eligible_positions else "N/A"
                current_pos = player.current_position if hasattr(player, 'current_position') and player.current_position else primary_pos
                eligible_positions = player.get_display_eligible_positions() if hasattr(player, 'get_display_eligible_positions') else [primary_pos]
                
                # ローラインを部分着色で描画
                is_current_pitcher = (player == fielding_team.current_pitcher)
                self._render_roster_line(
                    text_widget,
                    line_index=i + 1,
                    star=is_current_pitcher,
                    current_pos=current_pos,
                    player=player,
                    eligible_positions=eligible_positions,
                )
            else:
                self._clear_roster_line(text_widget)
    
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
                self.normal_batting_button.config(state=tk.NORMAL, text=self.text["normal_batting"])

        self._update_status_bar()
        self._update_toolbar_state()

    def _clear_roster_line(self, text_widget):
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.configure(state=tk.DISABLED)

    def _render_roster_line(self, text_widget, line_index: int, star: bool, current_pos: str, player, eligible_positions):
        """1行分のロスター表示を、ポジション語のみ色付きで描画する"""
        # 準備
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)

        # タグ（色）を確保
        def ensure_tag(color_name: str):
            tag_name = f"fg_{color_name}"
            try:
                text_widget.tag_configure(tag_name, foreground=color_name)
            except Exception:
                pass
            return tag_name

        # 先頭: 星印と番号
        prefix = ("★ " if star else "") + f"{line_index}. "
        text_widget.insert(tk.END, prefix)

        # 現在の守備位置（色付き）
        pos_color = get_position_color(current_pos, getattr(player, 'pitcher_type', None))
        if pos_color:
            tag = ensure_tag(pos_color)
            text_widget.insert(tk.END, current_pos, tag)
        else:
            text_widget.insert(tk.END, current_pos)

        # 区切りと名前
        text_widget.insert(tk.END, f" | {player.name} [Eligible: ")

        # Eligible positions（各トークンを色付け、区切りは通常色）
        for idx, pos in enumerate(eligible_positions):
            if idx > 0:
                text_widget.insert(tk.END, ", ")
            e_color = get_position_color(pos, getattr(player, 'pitcher_type', None) if pos in {"P", "SP", "RP"} else None)
            if e_color:
                tag = ensure_tag(e_color)
                text_widget.insert(tk.END, pos, tag)
            else:
                text_widget.insert(tk.END, pos)

        text_widget.insert(tk.END, "]")

        # 固定: 編集不可に戻す
        text_widget.configure(state=tk.DISABLED)
    
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
        elif color == "orange":
            self.result_text.tag_config("orange", foreground="orange", font=("Helvetica", 10, "bold"))
            self.result_text.insert(tk.END, text, "orange")
        else:
            self.result_text.insert(tk.END, text)
            
        self.result_text.config(state=tk.DISABLED)
        if self.log_auto_scroll.get():
            self.result_text.see(tk.END)
        self._update_toolbar_state()
    
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
                current_batter = self.game_state.batting_team.current_batter
                current_pitcher = self.game_state.fielding_team.current_pitcher
                from baseball_sim.gameplay.utils import BuntCalculator
                bunt_success_rate = BuntCalculator.calculate_bunt_success_probability(current_batter, current_pitcher)
                self.bunt_button.config(text=f"Bunt ({bunt_success_rate:.1%})")
            except Exception as e:
                logger.debug(f"Failed to compute bunt success rate: {e}")
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
            self._add_result_text("Please fix the defensive position issues before continuing.\n", "orange")
            return
        
        # 攻守交代が起きるかチェック（イニングの変わり目）
        prev_inning = self.game_state.inning
        prev_is_top = self.game_state.is_top_inning
        
        batter = self.game_state.batting_team.current_batter
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
            self._add_result_text(side_message, "orange")
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
            self._add_result_text("Please fix the defensive position issues before continuing.\n", "orange")
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
        
        batter = self.game_state.batting_team.current_batter
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
            self._add_result_text(side_message, "orange")
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

        if self.game_state:
            self.game_state.game_ended = True

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
        self._update_status_bar(self.text["status_game_over"])
        self._update_toolbar_state()

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
            self._add_result_text(side_message, "orange")
    
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
        home_team, away_team = self.team_manager.get_teams()
        if not home_team or not away_team:
            messagebox.showinfo("Info", "Teams are not ready yet.")
            return

        if not self.game_state:
            try:
                self.game_state = GameState(home_team, away_team)
            except Exception as exc:
                logger.error(f"Failed to start game: {exc}")
                messagebox.showerror("Error", f"Failed to start game: {exc}")
                return

        self._pending_log_reset = False
        self.event_manager.trigger(Events.GAME_STARTED, self.game_state)

        logger.info("Game started successfully with configured lineups!")

    def _show_team_preview(self, team_type):
        """チームプレビューを表示"""
        team = self.team_manager.get_team_by_type(team_type)
        if not team:
            messagebox.showinfo("Info", "Team is not ready yet.")
            return

        title_key = "team_preview_title_home" if team_type == 'home' else "team_preview_title_away"
        dialog = self.layout_manager.create_centered_dialog(self.text[title_key], width=640, height=520)

        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=12, pady=(12, 0))

        ttk.Label(header_frame, text=team.name, font=("TkDefaultFont", 14, "bold")).pack(side=tk.LEFT)

        status = self.team_manager.get_team_status(team_type)
        ttk.Label(header_frame, text=status['message'], foreground=status['color']).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ラインナップタブ
        lineup_tab = ttk.Frame(notebook)
        notebook.add(lineup_tab, text=self.text["team_preview_lineup"])
        lineup_tree = self._create_preview_table(
            lineup_tab,
            columns=("order", "position", "name", "eligible"),
            headings=(
                self.text["preview_order"],
                self.text["preview_position"],
                self.text["preview_name"],
                self.text["preview_eligible"],
            ),
            widths=(40, 70, 220, 240)
        )
        if team.lineup:
            for idx, player in enumerate(team.lineup, start=1):
                position = getattr(player, 'current_position', None) or getattr(player, 'position', '-')
                lineup_tree.insert("", tk.END, values=(idx, position, player.name, self._format_player_positions(player)))
        else:
            lineup_tree.insert("", tk.END, values=("", "", self.text["preview_no_players"], ""))

        # ベンチタブ
        bench_tab = ttk.Frame(notebook)
        notebook.add(bench_tab, text=self.text["team_preview_bench"])
        bench_tree = self._create_preview_table(
            bench_tab,
            columns=("position", "name", "eligible"),
            headings=(
                self.text["preview_position"],
                self.text["preview_name"],
                self.text["preview_eligible"],
            ),
            widths=(70, 240, 240)
        )
        bench_players = getattr(team, 'bench', [])
        if bench_players:
            for player in bench_players:
                pos = getattr(player, 'current_position', None) or getattr(player, 'position', '-')
                bench_tree.insert("", tk.END, values=(pos, player.name, self._format_player_positions(player)))
        else:
            bench_tree.insert("", tk.END, values=("", self.text["preview_no_players"], ""))

        # 投手タブ
        pitchers_tab = ttk.Frame(notebook)
        notebook.add(pitchers_tab, text=self.text["team_preview_pitchers"])
        pitchers_tree = self._create_preview_table(
            pitchers_tab,
            columns=("name", "role", "eligible"),
            headings=(
                self.text["preview_name"],
                self.text["preview_role"],
                self.text["preview_eligible"],
            ),
            widths=(220, 120, 210)
        )

        pitchers = list(getattr(team, 'pitchers', []))
        current_pitcher = getattr(team, 'current_pitcher', None)
        if current_pitcher and current_pitcher not in pitchers:
            pitchers.insert(0, current_pitcher)

        if pitchers:
            for player in pitchers:
                role = self.text["preview_role_current_pitcher"] if player == current_pitcher else self.text["preview_role_available"]
                pitchers_tree.insert("", tk.END, values=(player.name, role, self._format_player_positions(player)))
        else:
            pitchers_tree.insert("", tk.END, values=(self.text["preview_no_players"], "", ""))

        ttk.Button(dialog, text=self.text["close"], command=dialog.destroy).pack(pady=(0, 12))

    def _create_preview_table(self, parent, columns, headings, widths):
        """プレビュー用のテーブルを作成"""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tree = ttk.Treeview(container, columns=columns, show='headings', selectmode='none')
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

        for column, heading, width in zip(columns, headings, widths):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor=tk.W, stretch=True)

        return tree

    def _format_player_positions(self, player):
        """選手の守備可能ポジションを文字列化"""
        positions = []
        if hasattr(player, 'get_display_eligible_positions'):
            positions = player.get_display_eligible_positions()
        elif hasattr(player, 'eligible_positions') and player.eligible_positions:
            positions = player.eligible_positions

        if not positions:
            return "-"

        return ", ".join(positions)
    

    

    

    

    

    

    
    def _update_button_states(self):
        """ゲーム開始後のボタン状態を更新"""
        if not self.game_state:
            return
        
        # 通常打撃ボタンの状態を更新
        if hasattr(self, 'normal_batting_button'):
            self.normal_batting_button.config(state=tk.NORMAL, text=self.text["normal_batting"])

        # バントボタンの状態を更新
        self._update_bunt_button_state()
        
        # 戦略ボタンの状態を更新
        if hasattr(self, 'offense_button'):
            self.offense_button.config(state=tk.NORMAL)
        if hasattr(self, 'defense_button'):
            self.defense_button.config(state=tk.NORMAL)
        if hasattr(self, 'stats_button'):
            self.stats_button.config(state=tk.NORMAL)

        self._update_toolbar_state()
    
