"""エラーハンドリングとロギング機能"""

import logging
import traceback
from functools import wraps
from typing import Callable

from baseball_sim.config import get_project_paths

# ログファイルパスの設定
project_root = get_project_paths().project_root
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
