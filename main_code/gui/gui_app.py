"""
BaseballGUI Application Controller
GUIアプリケーションの中核となるコントローラークラス
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# プロジェクトルートの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .gui_main import BaseballGUI
from .gui_team_manager import TeamManager
from .gui_event_manager import EventManager
from .gui_layout_manager import LayoutManager


class BaseballGUIApp:
    """
    野球シミュレーションGUIアプリケーションのメインコントローラー
    """
    
    def __init__(self):
        self.root = None
        self.main_gui = None
        self.team_manager = None
        self.event_manager = None
        self.layout_manager = None
        
    def initialize(self):
        """アプリケーションを初期化"""
        # Tkinter rootの作成
        self.root = tk.Tk()
        self.root.title("Baseball Simulation")
        
        # マネージャーの初期化
        self.team_manager = TeamManager()
        self.event_manager = EventManager()
        self.layout_manager = LayoutManager(self.root)
        
        # メインGUIの初期化
        self.main_gui = BaseballGUI(
            root=self.root,
            team_manager=self.team_manager,
            event_manager=self.event_manager,
            layout_manager=self.layout_manager
        )
        
        # イベントハンドラーの設定
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """イベントハンドラーを設定"""
        # チーム作成完了イベント
        self.event_manager.bind('teams_created', self.main_gui.on_teams_created)
        
        # ゲーム開始イベント
        self.event_manager.bind('game_started', self.main_gui.on_game_started)
        
        # ラインナップ変更イベント
        self.event_manager.bind('lineup_changed', self.main_gui.on_lineup_changed)
        
    def run(self):
        """アプリケーションを実行"""
        if not self.root:
            self.initialize()
        
        self.main_gui.show_title_screen()
        self.root.mainloop()
        
    def shutdown(self):
        """アプリケーションを終了"""
        if self.root:
            self.root.quit()


def create_gui_app():
    """GUIアプリケーションを作成して返す"""
    app = BaseballGUIApp()
    app.initialize()
    return app


# 互換性のためのエイリアス
BaseballApp = BaseballGUIApp


if __name__ == "__main__":
    app = create_gui_app()
    app.run()
