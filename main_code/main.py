#!/usr/bin/env python3
"""
野球シミュレーションのメインエントリーポイント
リファクタリング後の構造でアプリケーションを起動
"""
import sys
import os

# プロジェクト設定を最初にロード（パス設定も含む）
# main.pyは直接実行されるため、絶対インポートを使用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_config import setup_project_environment
setup_project_environment()

from app_initializer import AppInitializer
from game_mode_manager import GameModeManager
from error_handling import logger


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
