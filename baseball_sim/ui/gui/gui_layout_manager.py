"""Layout management utilities for the Tkinter-based GUI."""

import tkinter as tk
from tkinter import ttk

from baseball_sim.config import UIConstants, setup_project_environment

from .gui_constants import get_font_settings, get_ui_text

setup_project_environment()


class LayoutManager:
    """
    GUI画面レイアウトの管理クラス
    画面の作成・切り替え・レイアウト調整を担当
    """
    
    def __init__(self, root):
        self.root = root
        self.main_frame = None
        self.current_screen = None
        
        # UI設定
        self.text = get_ui_text()
        self.default_font, self.title_font = get_font_settings()
        
        # 初期設定
        self._setup_window()
        self._create_main_frame()
    
    def _setup_window(self):
        """ウィンドウの基本設定"""
        self.root.geometry(f"{UIConstants.WINDOW_WIDTH}x{UIConstants.WINDOW_HEIGHT}")
        self.root.resizable(True, True)
        self.root.option_add("*Font", self.default_font)
    
    def _create_main_frame(self):
        """メインフレームを作成"""
        if self.main_frame:
            self.main_frame.destroy()
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
    
    def clear_screen(self):
        """現在の画面をクリア"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.current_screen = None
    
    def create_title_screen(self, on_new_game=None, on_exit=None, on_revalidate=None):
        """タイトル画面を作成
        
        Args:
            on_new_game: 新規ゲーム開始時のコールバック
            on_exit: 終了時のコールバック

        Returns:
            dict: 画面要素への参照を格納した辞書
        """
        self.clear_screen()
        self.current_screen = "title"

        elements = {}

        # タイトル
        title_label = ttk.Label(
            self.main_frame,
            text=self.text["title"],
            font=self.title_font
        )
        title_label.pack(pady=(40, 30))
        elements["title_label"] = title_label

        # ボタン枠
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=(0, 20))
        elements["button_frame"] = button_frame

        # スタートボタン
        start_button = None
        if on_new_game:
            start_button = ttk.Button(
                button_frame,
                text=self.text["new_game"],
                command=on_new_game,
                width=24
            )
            start_button.pack(pady=5)
        elements["start_button"] = start_button

        # ルールチェック（ラインナップ再バリデート）ボタン
        revalidate_button = None
        if on_revalidate:
            revalidate_button = ttk.Button(
                button_frame,
                text=self.text["revalidate_lineups"],
                command=on_revalidate,
                width=24
            )
            revalidate_button.pack(pady=5)
        elements["revalidate_button"] = revalidate_button

        # 終了ボタン
        quit_button = None
        if on_exit:
            quit_button = ttk.Button(
                button_frame,
                text=self.text["exit"],
                command=on_exit,
                width=24
            )
            quit_button.pack(pady=5)
        elements["quit_button"] = quit_button

        # チームステータス
        status_frame = ttk.LabelFrame(
            self.main_frame,
            text=self.text["team_status"]
        )
        status_frame.pack(fill=tk.X, padx=40, pady=(10, 20))
        elements["status_frame"] = status_frame

        status_rows = {}
        for idx, team_key in enumerate(("away", "home")):
            row_frame = ttk.Frame(status_frame)
            row_frame.pack(fill=tk.X, padx=10, pady=5)

            name_label = ttk.Label(row_frame, text="-", width=18, anchor=tk.W)
            name_label.pack(side=tk.LEFT)

            state_label = ttk.Label(row_frame, text=self.text["team_status_unknown"], anchor=tk.W)
            state_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

            manage_button = ttk.Button(
                row_frame,
                text=self.text["manage_team"],
                state=tk.DISABLED
            )
            manage_button.pack(side=tk.RIGHT, padx=(5, 0))

            detail_button = ttk.Button(
                row_frame,
                text=self.text["view_lineup"],
                state=tk.DISABLED
            )
            detail_button.pack(side=tk.RIGHT)

            status_rows[team_key] = {
                "frame": row_frame,
                "name_label": name_label,
                "state_label": state_label,
                "detail_button": detail_button,
                "manage_button": manage_button,
            }

        elements["status_rows"] = status_rows

        # ヒントテキスト
        hint_label = ttk.Label(
            self.main_frame,
            text=self.text["title_hint"],
            wraplength=520,
            justify=tk.CENTER
        )
        hint_label.pack(pady=(0, 10))
        elements["hint_label"] = hint_label

        return elements
    
    def create_game_screen_layout(self):
        """ゲーム画面のレイアウトを作成して返す
        
        Returns:
            dict: 画面の各領域の参照
        """
        self.clear_screen()
        self.current_screen = "game"
        
        # トップツールバー
        toolbar_frame = ttk.Frame(self.main_frame)
        toolbar_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # コンテンツ領域
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左側：フィールド表示エリア
        field_frame = ttk.Frame(content_frame)
        field_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # フィールドキャンバス
        field_canvas = tk.Canvas(
            field_frame, 
            width=UIConstants.FIELD_CANVAS_WIDTH,
            height=UIConstants.FIELD_CANVAS_HEIGHT, 
            bg="green"
        )
        field_canvas.pack(padx=10, pady=10)
        
        # 選手名簿表示エリア
        roster_frame = ttk.Frame(field_frame)
        roster_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP)
        
        # 攻撃側チーム名簿
        offense_roster_frame = ttk.LabelFrame(roster_frame, text="Offense Team")
        offense_roster_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 守備側チーム名簿
        defense_roster_frame = ttk.LabelFrame(roster_frame, text="Defense Team")
        defense_roster_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # スコアボード領域
        scoreboard_frame = ttk.Frame(field_frame)
        scoreboard_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 右側：情報表示エリアとコントロール
        info_frame = ttk.Frame(content_frame)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # 試合状況パネル
        situation_frame = ttk.Frame(info_frame)
        situation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 采配ボタン
        control_frame = ttk.LabelFrame(info_frame, text=self.text["strategy"])
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 試合進行ボタンフレーム
        play_frame = ttk.LabelFrame(control_frame, text="Batting Action")
        play_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 結果表示エリア
        result_frame = ttk.LabelFrame(info_frame, text=self.text["play_result"])
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ステータスバー
        status_bar_frame = ttk.Frame(self.main_frame)
        status_bar_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        status_label = ttk.Label(status_bar_frame, text="", anchor=tk.W)
        status_label.pack(side=tk.LEFT)

        action_label = ttk.Label(status_bar_frame, text="", anchor=tk.E)
        action_label.pack(side=tk.RIGHT)

        return {
            'toolbar_frame': toolbar_frame,
            'content_frame': content_frame,
            'field_frame': field_frame,
            'field_canvas': field_canvas,
            'offense_roster_frame': offense_roster_frame,
            'defense_roster_frame': defense_roster_frame,
            'scoreboard_frame': scoreboard_frame,
            'info_frame': info_frame,
            'situation_frame': situation_frame,
            'control_frame': control_frame,
            'play_frame': play_frame,
            'result_frame': result_frame,
            'status_bar_frame': status_bar_frame,
            'status_label': status_label,
            'action_label': action_label
        }
    
    def create_centered_dialog(self, title, width=600, height=500):
        """中央配置されたダイアログウィンドウを作成
        
        Args:
            title (str): ダイアログのタイトル
            width (int): 幅
            height (int): 高さ
            
        Returns:
            tk.Toplevel: 作成されたダイアログウィンドウ
        """
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry(f"{width}x{height}")
        dialog.grab_set()  # モーダルダイアログ
        
        # 親ウィンドウの中央に配置
        dialog.transient(self.root)
        
        # 位置の調整
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        return dialog
    
    def get_current_screen(self):
        """現在の画面タイプを取得"""
        return self.current_screen
    
    def get_main_frame(self):
        """メインフレームを取得"""
        return self.main_frame
