"""Main GUI class for the Baseball Simulation."""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from baseball_sim.config import GameResults, UIConstants, setup_project_environment
from baseball_sim.infrastructure.logging_utils import logger

from baseball_sim.gameplay.game import GameState
from baseball_sim.gameplay.lineup import LineupManager

from .gui_constants import get_ui_text
from .gui_event_manager import Events
from .gui_field import FieldManager
from .gui_scoreboard import ScoreboardManager
from .gui_stats import StatsManager
from .gui_strategy import StrategyManager
from .gui_colors import get_position_color
from .gui_defense_mode import DefenseChangeMode

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

    # --- Helper for position display ---
    def _display_position_token(self, pos: str, player) -> str:
        """Return the display token for a player's position.

        - For pitchers with position 'P', show 'SP' or 'RP' based on pitcher_type.
        - Otherwise return the original token.
        """
        if not pos:
            return pos
        if pos.upper() == "P" and hasattr(player, 'pitcher_type') and player.pitcher_type in ("SP", "RP"):
            return player.pitcher_type
        return pos

    def _render_eligible_tokens(self, parent, player):
        """Eligible欄を各ポジション単語のみ色付けして描画"""
        # 既存の子ウィジェットをクリア
        for child in parent.winfo_children():
            child.destroy()
        # 位置リストの取得（P→SP/RP 変換）
        positions = []
        if hasattr(player, 'get_display_eligible_positions'):
            positions = player.get_display_eligible_positions()
        elif hasattr(player, 'eligible_positions') and player.eligible_positions:
            positions = list(player.eligible_positions)
        # 配置
        if not positions:
            ttk.Label(parent, text="-").pack(side=tk.LEFT)
            return
        for idx, raw in enumerate(positions):
            token = self._display_position_token(raw, player)
            color = get_position_color(token, getattr(player, 'pitcher_type', None) if token in {"P", "SP", "RP"} else None)
            lbl = ttk.Label(parent, text=token)
            if color:
                try:
                    lbl.configure(foreground=color)
                except Exception:
                    pass
            lbl.pack(side=tk.LEFT)
            if idx < len(positions) - 1:
                ttk.Label(parent, text=", ").pack(side=tk.LEFT)
    
    def _setup_event_listeners(self):
        """イベントリスナーを設定"""
        self.event_manager.bind(Events.TEAMS_CREATED, self.on_teams_created)
        self.event_manager.bind(Events.GAME_STARTED, self.on_game_started)
        self.event_manager.bind(Events.LINEUP_CHANGED, self.on_lineup_changed)
        self.event_manager.bind(Events.GAME_STATE_CHANGED, self.on_game_state_changed)
    
    def show_title_screen(self):
        """タイトル画面を表示"""
        self._unbind_keyboard_shortcuts()
        # タイトル画面表示前にチームをロードして、プレビュー/バリデーションを使えるようにする
        # 通知は不要（イベント再帰を避けるため）
        self._load_teams(force_reload=False, notify=False)

        self.title_screen_elements = self.layout_manager.create_title_screen(
            on_new_game=self._start_game,
            on_exit=self.root.quit,
            on_revalidate=self._revalidate_lineups
        )

        status_rows = self.title_screen_elements.get("status_rows", {})
        if "home" in status_rows:
            status_rows["home"]["detail_button"].config(
                command=lambda: self._show_team_preview('home')
            )
            status_rows["home"].get("manage_button").config(
                command=lambda: self._open_team_management('home')
            )
        if "away" in status_rows:
            status_rows["away"]["detail_button"].config(
                command=lambda: self._show_team_preview('away')
            )
            status_rows["away"].get("manage_button").config(
                command=lambda: self._open_team_management('away')
            )

        self._update_title_screen_status()

    def _revalidate_lineups(self):
        """タイトル画面からラインナップのバリデーションを再実行"""
        # チーム未ロードならロード（通知なし）
        if not self.team_manager.has_teams():
            if not self._load_teams(force_reload=False, notify=False):
                return
        # ステータスの再計算と反映
        self._update_title_screen_status()
        # 追加の通知は不要。視覚的更新で十分。

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
                manage_btn = widgets.get("manage_button")
                if manage_btn:
                    manage_btn.config(state=tk.NORMAL)

                if not status['valid'] and status['errors']:
                    invalid_messages.append(f"{team.name}: {len(status['errors'])}")
            else:
                widgets["name_label"].config(text=f"- ({label_suffix})")
                widgets["state_label"].config(text=self.text["team_status_unknown"], foreground="grey")
                widgets["detail_button"].config(state=tk.DISABLED)
                manage_btn = widgets.get("manage_button")
                if manage_btn:
                    manage_btn.config(state=tk.DISABLED)

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
        # タイトル画面で既にロード済みの場合が多いので通知は行わない
        if not self._load_teams(notify=False):
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
        display_pos = self._display_position_token(current_pos, player)
        pos_color = get_position_color(display_pos, getattr(player, 'pitcher_type', None))
        if pos_color:
            tag = ensure_tag(pos_color)
            text_widget.insert(tk.END, display_pos, tag)
        else:
            text_widget.insert(tk.END, display_pos)

        # 区切りと名前
        text_widget.insert(tk.END, f" | {player.name} [Eligible: ")

        # Eligible positions（各トークンを色付け、区切りは通常色）
        for idx, pos in enumerate(eligible_positions):
            if idx > 0:
                text_widget.insert(tk.END, ", ")
            display_epos = self._display_position_token(pos, player)
            e_color = get_position_color(display_epos, getattr(player, 'pitcher_type', None) if display_epos in {"P", "SP", "RP"} else None)
            if e_color:
                tag = ensure_tag(e_color)
                text_widget.insert(tk.END, display_epos, tag)
            else:
                text_widget.insert(tk.END, display_epos)

        text_widget.insert(tk.END, "]")

        # 固定: 編集不可に戻す
        text_widget.configure(state=tk.DISABLED)

    def _render_roster_text_block(self, text_widget, players, star_players=None):
        """複数選手を1つのTextウィジェットに描画する（ポジション語のみ着色）"""
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)

        if not players:
            text_widget.insert(tk.END, "-")
            text_widget.configure(state=tk.DISABLED)
            return

        star_players = set(filter(None, star_players or set()))

        def ensure_tag(color_name: str):
            tag_name = f"fg_{color_name}"
            if tag_name not in text_widget.tag_names():
                try:
                    text_widget.tag_configure(tag_name, foreground=color_name)
                except Exception:
                    pass
            return tag_name

        for idx, player in enumerate(players, start=1):
            prefix = ("★ " if player in star_players else "") + f"{idx}. "
            text_widget.insert(tk.END, prefix)

            eligible_positions = []
            if hasattr(player, 'get_display_eligible_positions'):
                eligible_positions = list(player.get_display_eligible_positions() or [])
            elif hasattr(player, 'eligible_positions') and player.eligible_positions:
                eligible_positions = list(player.eligible_positions)

            primary_pos = eligible_positions[0] if eligible_positions else None
            fallback_pos = getattr(player, 'position', None)
            current_pos = getattr(player, 'current_position', None) or primary_pos or fallback_pos or "-"
            display_pos = self._display_position_token(current_pos, player)
            pos_color = get_position_color(display_pos, getattr(player, 'pitcher_type', None))
            if pos_color:
                tag = ensure_tag(pos_color)
                text_widget.insert(tk.END, display_pos, tag)
            else:
                text_widget.insert(tk.END, display_pos)

            text_widget.insert(tk.END, f" | {player.name}")

            if not eligible_positions and current_pos and current_pos != "-":
                eligible_positions = [current_pos]

            if eligible_positions:
                text_widget.insert(tk.END, " [Eligible: ")
                for e_idx, pos in enumerate(eligible_positions):
                    if e_idx > 0:
                        text_widget.insert(tk.END, ", ")
                    display_epos = self._display_position_token(pos, player)
                    e_color = get_position_color(
                        display_epos,
                        getattr(player, 'pitcher_type', None) if display_epos in {"P", "SP", "RP"} else None
                    )
                    if e_color:
                        tag = ensure_tag(e_color)
                        text_widget.insert(tk.END, display_epos, tag)
                    else:
                        text_widget.insert(tk.END, display_epos)
                text_widget.insert(tk.END, "]")

            text_widget.insert(tk.END, "\n")

        text_widget.delete("end-1c", tk.END)
        text_widget.configure(state=tk.DISABLED)
        try:
            text_widget.yview_moveto(0.0)
        except Exception:
            pass

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

        # ラインナップタブ（セル内部分色のため、ラベルグリッドで自前描画）
        lineup_tab = ttk.Frame(notebook)
        notebook.add(lineup_tab, text=self.text["team_preview_lineup"])
        # ヘッダー
        header = ttk.Frame(lineup_tab)
        header.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Label(header, text=self.text["preview_order"], width=4).grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Label(header, text=self.text["preview_position"], width=6).grid(row=0, column=1, sticky=tk.W, padx=2)
        ttk.Label(header, text=self.text["preview_name"], width=28).grid(row=0, column=2, sticky=tk.W, padx=2)
        ttk.Label(header, text=self.text["preview_eligible"], width=34).grid(row=0, column=3, sticky=tk.W, padx=2)
        # 行
        rows = ttk.Frame(lineup_tab)
        rows.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        if team.lineup:
            for i, player in enumerate(team.lineup, start=1):
                raw_pos = getattr(player, 'current_position', None) or getattr(player, 'position', '-')
                position = self._display_position_token(raw_pos, player)
                color = get_position_color(position, getattr(player, 'pitcher_type', None))
                ttk.Label(rows, text=str(i), width=4).grid(row=i, column=0, sticky=tk.W, padx=2, pady=1)
                pos_lbl = ttk.Label(rows, text=position, width=6)
                if color:
                    try:
                        pos_lbl.configure(foreground=color)
                    except Exception:
                        pass
                pos_lbl.grid(row=i, column=1, sticky=tk.W, padx=2, pady=1)
                ttk.Label(rows, text=player.name, width=28).grid(row=i, column=2, sticky=tk.W, padx=2, pady=1)
                # Eligible を各トークン色付けで描画
                elig_container = ttk.Frame(rows)
                elig_container.grid(row=i, column=3, sticky=tk.W, padx=2, pady=1)
                self._render_eligible_tokens(elig_container, player)
        else:
            ttk.Label(rows, text=self.text["preview_no_players"]).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=2, pady=2)

        # ベンチタブ（ラベルグリッドで部分色）
        bench_tab = ttk.Frame(notebook)
        notebook.add(bench_tab, text=self.text["team_preview_bench"])
        b_header = ttk.Frame(bench_tab)
        b_header.pack(fill=tk.X, padx=8, pady=(6, 2))
        # ベンチはポジション列を表示しない（要望に合わせて非表示）
        ttk.Label(b_header, text=self.text["preview_name"], width=28).grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Label(b_header, text=self.text["preview_eligible"], width=34).grid(row=0, column=1, sticky=tk.W, padx=2)
        b_rows = ttk.Frame(bench_tab)
        b_rows.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        bench_players = getattr(team, 'bench', [])
        if bench_players:
            for i, player in enumerate(bench_players, start=1):
                ttk.Label(b_rows, text=player.name, width=28).grid(row=i, column=0, sticky=tk.W, padx=2, pady=1)
                # Eligible を各トークン色付けで描画
                elig_container = ttk.Frame(b_rows)
                elig_container.grid(row=i, column=1, sticky=tk.W, padx=2, pady=1)
                self._render_eligible_tokens(elig_container, player)
        else:
            ttk.Label(b_rows, text=self.text["preview_no_players"]).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=2, pady=2)

        # 投手タブ（Eligible の各トークンを色付けするためラベルグリッドで描画）
        pitchers_tab = ttk.Frame(notebook)
        notebook.add(pitchers_tab, text=self.text["team_preview_pitchers"])
        p_header = ttk.Frame(pitchers_tab)
        p_header.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Label(p_header, text=self.text["preview_name"], width=28).grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Label(p_header, text=self.text["preview_role"], width=14).grid(row=0, column=1, sticky=tk.W, padx=2)
        ttk.Label(p_header, text=self.text["preview_eligible"], width=34).grid(row=0, column=2, sticky=tk.W, padx=2)
        p_rows = ttk.Frame(pitchers_tab)
        p_rows.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        pitchers = list(getattr(team, 'pitchers', []))
        current_pitcher = getattr(team, 'current_pitcher', None)
        if current_pitcher and current_pitcher not in pitchers:
            pitchers.insert(0, current_pitcher)

        if pitchers:
            for i, player in enumerate(pitchers, start=1):
                role = self.text["preview_role_current_pitcher"] if player == current_pitcher else self.text["preview_role_available"]
                # 名前
                ttk.Label(p_rows, text=player.name, width=28).grid(row=i, column=0, sticky=tk.W, padx=2, pady=1)
                # 役割（必要なら色付け可能だが、今回はそのまま）
                ttk.Label(p_rows, text=role, width=14).grid(row=i, column=1, sticky=tk.W, padx=2, pady=1)
                # Eligible（SP/RP を色付け）
                elig_container = ttk.Frame(p_rows)
                elig_container.grid(row=i, column=2, sticky=tk.W, padx=2, pady=1)
                self._render_eligible_tokens(elig_container, player)
        else:
            ttk.Label(p_rows, text=self.text["preview_no_players"]).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=2, pady=2)

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

    def _open_team_management(self, team_type):
        """スタート画面からチーム設定ダイアログを開く"""
        team = self.team_manager.get_team_by_type(team_type)
        if not team:
            messagebox.showinfo("Info", "Team is not ready yet.")
            return

        title_key = "team_manage_title_home" if team_type == 'home' else "team_manage_title_away"
        dialog = self.layout_manager.create_centered_dialog(self.text[title_key], width=720, height=620)

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text=team.name, font=("TkDefaultFont", 14, "bold")).pack(side=tk.LEFT)
        status_label = ttk.Label(header, text="", foreground="green")
        status_label.pack(side=tk.RIGHT)

        lineup_frame = ttk.LabelFrame(main_frame, text=self.text["team_manage_lineup"])
        lineup_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 6))
        lineup_display = scrolledtext.ScrolledText(lineup_frame, height=10, wrap=tk.NONE)
        lineup_display.configure(state=tk.DISABLED)
        lineup_display.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        bench_frame = ttk.LabelFrame(main_frame, text=self.text["team_manage_bench"])
        bench_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        bench_display = scrolledtext.ScrolledText(bench_frame, height=7, wrap=tk.NONE)
        bench_display.configure(state=tk.DISABLED)
        bench_display.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        pitcher_frame = ttk.LabelFrame(main_frame, text=self.text["team_manage_pitcher"])
        pitcher_frame.pack(fill=tk.X, pady=(0, 8))
        pitcher_label = ttk.Label(pitcher_frame, text="-")
        pitcher_label.pack(anchor=tk.W, padx=8, pady=6)

        controls = ttk.Frame(main_frame)
        controls.pack(fill=tk.X, pady=(4, 6))

        def refresh_lists():
            lineup_players = list(team.lineup)
            current_pitcher = getattr(team, 'current_pitcher', None)
            highlight_players = {current_pitcher} if current_pitcher in lineup_players else set()
            self._render_roster_text_block(lineup_display, lineup_players, highlight_players)

            bench_players = list(team.bench)
            self._render_roster_text_block(bench_display, bench_players)

            current_pitcher = getattr(team, 'current_pitcher', None)
            if current_pitcher:
                info = f"{current_pitcher.name}"
                if hasattr(current_pitcher, 'pitcher_type'):
                    info += f" ({current_pitcher.pitcher_type})"
                pitcher_label.config(text=info)
            else:
                pitcher_label.config(text="-")

            status = self.team_manager.get_team_status(team_type)
            status_label.config(text=status['message'], foreground=status['color'])

        def open_defense_setup():
            substitution_manager = self.team_manager.get_substitution_manager(team_type)
            if not substitution_manager:
                messagebox.showerror("Error", "Substitution manager not available")
                return
            defense_mode = DefenseChangeMode(dialog, substitution_manager, self, allow_retire=False)
            defense_mode.start_defense_change_mode()
            if defense_mode.mode_window:
                self.root.wait_window(defense_mode.mode_window)
            refresh_lists()
            self._update_title_screen_status()
            self.event_manager.trigger(Events.LINEUP_CHANGED, team_type)

        def open_swap_dialog():
            self._open_batting_order_swap_dialog(team_type, dialog, refresh_lists)

        def open_pitcher_dialog():
            self._open_pitcher_selection_dialog(team_type, dialog, refresh_lists)

        ttk.Button(controls, text=self.text["team_manage_defense"], command=open_defense_setup).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text=self.text["team_manage_order"], command=open_swap_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text=self.text["team_manage_pitcher_button"], command=open_pitcher_dialog).pack(side=tk.LEFT, padx=4)

        ttk.Button(main_frame, text=self.text["team_manage_close"], command=dialog.destroy).pack(anchor=tk.E, pady=(8, 0))

        refresh_lists()

    def _open_batting_order_swap_dialog(self, team_type, parent, refresh_callback):
        """打順の入れ替えダイアログを表示"""
        team = self.team_manager.get_team_by_type(team_type)
        if not team:
            messagebox.showinfo("Info", "Team is not ready yet.")
            return

        lineup_manager = self.team_manager.get_lineup_manager(team_type)
        if not lineup_manager:
            lineup_manager = LineupManager(team)

        dialog = tk.Toplevel(parent)
        dialog.title(self.text["order_swap_title"].format(team.name))
        dialog.geometry("420x460")
        dialog.transient(parent)
        dialog.grab_set()

        ttk.Label(dialog, text=self.text["order_swap_instruction"]).pack(pady=8)

        listbox = tk.Listbox(dialog, selectmode=tk.MULTIPLE, height=12)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def refresh_list():
            listbox.delete(0, tk.END)
            for idx, player in enumerate(team.lineup, start=1):
                position = getattr(player, 'current_position', getattr(player, 'position', '-'))
                listbox.insert(tk.END, f"{idx}. {player.name} ({position})")

        def on_swap():
            selection = listbox.curselection()
            if len(selection) != 2:
                messagebox.showinfo("Info", self.text["order_swap_error_selection"])
                return
            idx1, idx2 = selection
            name1 = team.lineup[idx1].name
            name2 = team.lineup[idx2].name
            success, message = lineup_manager.swap_players_in_batting_order(idx1, idx2)
            if success:
                refresh_list()
                refresh_callback()
                self._update_title_screen_status()
                self.event_manager.trigger(Events.LINEUP_CHANGED, team_type)
                messagebox.showinfo("Success", self.text["order_swap_success"].format(name1, name2))
            else:
                messagebox.showerror("Error", message)

        ttk.Button(dialog, text=self.text["team_manage_order"], command=on_swap).pack(pady=6)
        ttk.Button(dialog, text=self.text["close"], command=dialog.destroy).pack(pady=(0, 8))

        refresh_list()

    def _open_pitcher_selection_dialog(self, team_type, parent, refresh_callback):
        """先発投手選択ダイアログを表示"""
        team = self.team_manager.get_team_by_type(team_type)
        if not team:
            messagebox.showinfo("Info", "Team is not ready yet.")
            return

        dialog = tk.Toplevel(parent)
        dialog.title(self.text["pitcher_select_title"].format(team.name))
        dialog.geometry("360x420")
        dialog.transient(parent)
        dialog.grab_set()

        ttk.Label(dialog, text=self.text["pitcher_select_instruction"]).pack(pady=10)

        listbox = tk.Listbox(dialog, selectmode=tk.SINGLE, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def build_pitcher_list():
            pitchers = []
            current = getattr(team, 'current_pitcher', None)
            if current:
                pitchers.append(current)
            for pitcher in getattr(team, 'pitchers', []):
                if pitcher not in pitchers:
                    pitchers.append(pitcher)
            return pitchers

        pitcher_objects = build_pitcher_list()

        def refresh_listbox():
            nonlocal pitcher_objects
            pitcher_objects = build_pitcher_list()
            listbox.delete(0, tk.END)
            for pitcher in pitcher_objects:
                label = pitcher.name
                if hasattr(pitcher, 'pitcher_type'):
                    label += f" ({pitcher.pitcher_type})"
                listbox.insert(tk.END, label)

        def on_apply():
            selection = listbox.curselection()
            if not selection:
                messagebox.showinfo("Info", "Please select a pitcher.")
                return
            idx = selection[0]
            selected = pitcher_objects[idx]
            current = getattr(team, 'current_pitcher', None)
            if selected is current:
                messagebox.showinfo("Info", self.text["pitcher_set_same"].format(selected.name))
                dialog.destroy()
                return

            if selected in team.pitchers:
                team.pitchers.remove(selected)
            if current and current not in team.pitchers:
                team.pitchers.append(current)
            team.current_pitcher = selected

            messagebox.showinfo("Success", self.text["pitcher_set_success"].format(selected.name))
            refresh_callback()
            self._update_title_screen_status()
            self.event_manager.trigger(Events.LINEUP_CHANGED, team_type)
            dialog.destroy()

        ttk.Button(dialog, text=self.text["team_manage_pitcher_button"], command=on_apply).pack(pady=6)
        ttk.Button(dialog, text=self.text["close"], command=dialog.destroy).pack(pady=(0, 8))

        refresh_listbox()

    def _format_player_positions(self, player):
        """選手の守備可能ポジションを文字列化"""
        positions = []
        if hasattr(player, 'get_display_eligible_positions'):
            positions = player.get_display_eligible_positions()
        elif hasattr(player, 'eligible_positions') and player.eligible_positions:
            positions = player.eligible_positions

        if not positions:
            return "-"

        # 表示トークン（P→SP/RP 変換）
        mapped = [self._display_position_token(p, player) for p in positions]
        return ", ".join(mapped)
    

    

    

    

    

    

    
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
    
