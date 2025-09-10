"""
エラーハンドリングとロギング機能
"""
import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps
from pathlib import Path

# ログファイルパスの設定
project_root = Path(__file__).parent.parent
log_file_path = project_root / "simulation_results" / "baseball_sim.log"

# ログディレクトリが存在しない場合は作成
log_file_path.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('baseball_sim')

class BaseballSimError(Exception):
    """ベースボールシミュレーションの基底例外クラス"""
    pass

class GameStateError(BaseballSimError):
    """ゲーム状態に関するエラー"""
    pass

class LineupError(BaseballSimError):
    """ラインナップに関するエラー"""
    pass

class PlayerError(BaseballSimError):
    """選手に関するエラー"""
    pass

class DataLoadError(BaseballSimError):
    """データ読み込みに関するエラー"""
    pass

def log_error(func: Callable) -> Callable:
    """エラーログを記録するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    return wrapper

def handle_exceptions(default_return: Any = None, 
                     exception_types: tuple = (Exception,),
                     log_level: str = "error") -> Callable:
    """例外処理を行うデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func(f"Exception in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator

class ErrorReporter:
    """エラーレポート機能"""
    
    @staticmethod
    def report_validation_errors(errors: list, context: str = ""):
        """バリデーションエラーを報告"""
        if not errors:
            return
        
        error_msg = f"Validation errors{f' in {context}' if context else ''}:"
        logger.warning(error_msg)
        for error in errors:
            logger.warning(f"  - {error}")
    
    @staticmethod
    def report_lineup_issues(team_name: str, issues: list):
        """ラインナップの問題を報告"""
        if not issues:
            logger.info(f"Lineup for {team_name} is valid")
            return
        
        logger.warning(f"Lineup issues for {team_name}:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    @staticmethod
    def report_game_state(game_state, action: str):
        """ゲーム状態を報告"""
        logger.info(f"{action}: Inning {game_state.inning}, "
                   f"{'Top' if game_state.is_top_inning else 'Bottom'}, "
                   f"Score: {game_state.away_score}-{game_state.home_score}")
