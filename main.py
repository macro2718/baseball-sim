#!/usr/bin/env python3
"""
野球シミュレーションのメインエントリーポイント
リファクタリング後の構造でアプリケーションを起動
"""
import sys
from app_initializer import AppInitializer
from game_mode_manager import GameModeManager
from error_handling import logger

# 後方互換性のためのラッパー関数
def create_sample_teams():
    """レガシーサポート: DataLoaderを使用してチームを作成"""
    from data_loader import DataLoader
    return DataLoader.create_teams_from_data()

def play_game_gui(home_team=None, away_team=None):
    """レガシーサポート: GUIゲームを起動"""
    return GameModeManager.play_game_gui(home_team, away_team)

def get_game_mode_choice():
    """レガシーサポート: ゲームモード選択"""
    return GameModeManager.get_game_mode_choice()

def initialize_random_seed():
    """レガシーサポート: ランダムシード初期化"""
    AppInitializer.initialize_random_seed()


def main():
    """メイン関数"""
    try:
        # アプリケーション初期化
        if not AppInitializer.initialize_application():
            logger.error("Application initialization failed")
            sys.exit(1)
        
        # ゲームモード選択
        mode = GameModeManager.get_game_mode_choice()
        
        # ゲーム実行
        GameModeManager.execute_game_mode(mode)
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        print("\nプログラムが中断されました")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
