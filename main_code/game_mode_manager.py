"""
ゲームモード管理モジュール
ゲームの起動モードと実行ロジックを管理
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from config import config
from player_data.data_loader import DataLoader
from simulation import simulate_games
from terminal_mode import play_game_terminal


class GameModeManager:
    """ゲームモードの選択と実行を管理するクラス"""
    
    MODES = {
        "1": "terminal",
        "2": "gui",
        "3": "simulation",
        "0": None
    }
    
    @staticmethod
    def display_mode_selection() -> None:
        """モード選択画面を表示"""
        print("========== 野球シミュレーションモード選択 ==========")
        print("1: ターミナルモード - テキストベースのインターフェース (スタメン設定可能)")
        print("2: GUIモード - グラフィカルインターフェース (スタメン設定可能)")
        print("3: シミュレーションモード - 複数試合の自動シミュレーション")
        print("0: 終了")
    
    @classmethod
    def get_game_mode_choice(cls) -> Optional[str]:
        """ユーザーからゲームモードの選択を取得"""
        cls.display_mode_selection()
        choice = input("実行するモードを選択してください (0-3): ")
        
        mode = cls.MODES.get(choice)
        if mode is None and choice == "0":
            print("プログラムを終了します")
            sys.exit(0)
        elif mode is None:
            print("無効な選択です。デフォルトのGUIモードで起動します")
            mode = "gui"
        
        return mode
    
    @staticmethod
    def play_game_gui(home_team=None, away_team=None):
        """GUI-based game loop"""
        from gui.gui_app import BaseballApp
        
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
        num_games = int(input("シミュレーションする試合数を入力してください: "))
        output_file = input("結果を出力するファイル名を入力してください (デフォルト: タイムスタンプ付きファイル): ")
        if not output_file.strip():
            output_file = None  # Noneを渡すことでsimulate_games内でタイムスタンプ付きファイル名を生成
        return simulate_games(num_games, output_file)
    
    @classmethod
    def execute_game_mode(cls, mode: str):
        """指定されたモードでゲームを実行"""
        if mode == "gui":
            # GUIモードの場合、チーム作成はGUI内で行う
            return cls.play_game_gui()
        
        elif mode in ["terminal", "simulation"]:
            # ターミナル・シミュレーションモードの場合は事前にチームを作成
            home_team, away_team = DataLoader.create_teams_from_data()
            
            if mode == "terminal":
                return play_game_terminal(home_team, away_team)
            elif mode == "simulation":
                return cls.play_simulation_mode()
        
        else:
            raise ValueError(f"Unknown game mode: {mode}")
