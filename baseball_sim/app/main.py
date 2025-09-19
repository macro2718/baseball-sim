"""Application entry point for the baseball simulator."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from baseball_sim.config import setup_project_environment
from baseball_sim.infrastructure.initializer import AppInitializer
from baseball_sim.infrastructure.logging_utils import logger
from baseball_sim.interface.modes import GameModeManager


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
    setup_project_environment()
    main()
