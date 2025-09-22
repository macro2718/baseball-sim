"""Game mode selection and execution helpers."""

import sys

from baseball_sim.config import setup_project_environment
from baseball_sim.ui.gui.gui_app import BaseballApp
from baseball_sim.ui.gui.gui_startup import SimulationWindow, StartupWindow

setup_project_environment()


class GameModeManager:
    """ゲームモードの選択と実行を管理するクラス"""

    VALID_MODES = {"gui", "simulation"}

    @classmethod
    def get_game_mode_choice(cls) -> str:
        """モード選択ウィンドウを表示して選択されたモードを返す"""
        selector = StartupWindow()
        mode = selector.show()

        if mode not in cls.VALID_MODES:
            sys.exit(0)

        return mode
    
    @staticmethod
    def play_game_gui(home_team=None, away_team=None):
        """GUI-based game loop"""
        app = BaseballApp()
        app.initialize()
        
        # チームが事前に提供されている場合は設定
        if home_team and away_team:
            app.main_gui.set_teams(home_team, away_team)
        
        # アプリケーションを実行
        app.run()
        return None
    
    @staticmethod
    def play_simulation_mode():
        """シミュレーションモードの実行"""
        window = SimulationWindow()
        window.show()
        return None
    
    @classmethod
    def execute_game_mode(cls, mode: str):
        """指定されたモードでゲームを実行"""
        if mode == "gui":
            # GUIモードの場合、チーム作成はGUI内で行う
            return cls.play_game_gui()
        
        elif mode == "simulation":
            return cls.play_simulation_mode()

        else:
            raise ValueError(f"Unknown game mode: {mode}")
