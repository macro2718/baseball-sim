"""
設定管理クラス
アプリケーション全体の設定を一元管理
"""
import os
import json
from typing import Dict, Any
from .constants import FilePaths


class ConfigManager:
    """設定情報を一元管理するクラス"""
    
    def __init__(self):
        self._config = self._load_default_config()
        self._load_user_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を読み込み"""
        return {
            "game": {
                "max_innings": 12,
                "enable_dh": True,
                "mercy_rule_enabled": False,
                "mercy_rule_runs": 10
            },
            "ui": {
                "window_width": 1280,
                "window_height": 800,
                "field_width": 500,
                "field_height": 400,
                "default_language": "ja"
            },
            "simulation": {
                "default_games": 100,
                "use_ml_prediction": True,
                "random_seed": None,
                "prediction_model_type": "nn"  # "linear" または "nn"
            },
            "files": {
                "data_dir": FilePaths.DATA_DIR,
                "models_dir": FilePaths.MODELS_DIR,
                "players_file": FilePaths.PLAYERS_JSON,
                "teams_file": FilePaths.TEAMS_JSON
            }
        }
    
    def _load_user_config(self):
        """ユーザー設定ファイルを読み込み（存在する場合）"""
        # config.json をこのファイルと同じディレクトリから読み込む
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._merge_config(user_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load user config: {e}")
    
    def _merge_config(self, user_config: Dict[str, Any]):
        """ユーザー設定をデフォルト設定にマージ"""
        def merge_dict(default: dict, user: dict):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dict(default[key], value)
                else:
                    default[key] = value
        
        merge_dict(self._config, user_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得（ドット記法対応）"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """設定値を設定（ドット記法対応）"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_user_config(self, config_path: str = None):
        """ユーザー設定をファイルに保存"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving config: {e}")


# シングルトンインスタンス
config = ConfigManager()

