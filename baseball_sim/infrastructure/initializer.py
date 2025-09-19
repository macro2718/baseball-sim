"""Application bootstrap helpers."""

import random

from baseball_sim.config import config
from baseball_sim.infrastructure.logging_utils import logger


class AppInitializer:
    """アプリケーションの初期化を管理するクラス"""
    
    @staticmethod
    def initialize_random_seed() -> None:
        """ランダムシードを初期化"""
        seed = config.get('simulation.random_seed')
        if seed is not None:
            random.seed(seed)
            logger.info(f"Random seed set to: {seed}")
        else:
            random.seed(0)  # デフォルトシード
            logger.info("Random seed set to default: 0")
    
    @staticmethod
    def initialize_logging() -> None:
        """ログシステムを初期化"""
        # 必要に応じてログレベルやハンドラーを設定
        logger.info("Application logging initialized")
    
    @staticmethod
    def validate_system_requirements() -> bool:
        """システム要件をチェック"""
        try:
            # 必要な設定ファイルの存在確認
            required_config_keys = [
                'game.max_innings',
                'simulation.use_ml_prediction',
                'files.data_dir'
            ]
            
            for key in required_config_keys:
                if config.get(key) is None:
                    logger.warning(f"Missing required config: {key}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False
    
    @classmethod
    def initialize_application(cls) -> bool:
        """アプリケーション全体を初期化"""
        try:
            logger.info("Starting application initialization...")
            
            # システム要件チェック
            if not cls.validate_system_requirements():
                logger.error("System requirements validation failed")
                return False
            
            # ログシステム初期化
            cls.initialize_logging()
            
            # ランダムシード初期化
            cls.initialize_random_seed()
            
            logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return False
