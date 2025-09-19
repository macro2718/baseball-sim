"""Utility helpers for dealing with project file-system paths."""

import json
import os
from pathlib import Path
from typing import Any

from .constants import FilePaths
from .environment import get_project_paths


class FileUtils:
    """ファイル操作のユーティリティクラス"""
    
    @staticmethod
    def safe_json_load(filepath: str, default: Any = None) -> Any:
        """安全なJSON読み込み（エラー時はデフォルト値を返す）"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            return default
    
    @staticmethod
    def safe_json_save(data: Any, filepath: str) -> bool:
        """安全なJSON保存（成功時True、失敗時Falseを返す）"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, TypeError) as e:
            print(f"Error saving JSON to {filepath}: {e}")
            return False


class PathManager:
    """ファイルパス管理クラス"""
    
    def __init__(self, base_dir: str | None = None):
        project_root = get_project_paths().project_root
        self.base_dir = str(base_dir) if base_dir is not None else str(project_root)
    
    def get_data_path(self, filename: str = None) -> str:
        """player_data/dataディレクトリのパスを取得"""
        data_dir = os.path.join(self.base_dir, "player_data", "data")
        if filename:
            return os.path.join(data_dir, filename)
        return data_dir
    
    def get_models_path(self, filename: str = None) -> str:
        """prediction_models/modelsディレクトリのパスを取得"""
        models_dir = os.path.join(self.base_dir, "prediction_models", "models")
        if filename:
            return os.path.join(models_dir, filename)
        return models_dir
    
    def get_players_data_path(self) -> str:
        """選手データファイルのパスを取得"""
        return self.get_data_path(FilePaths.PLAYERS_JSON)
    
    def get_teams_data_path(self) -> str:
        """チームデータファイルのパスを取得"""
        return self.get_data_path(FilePaths.TEAMS_JSON)
    
    def get_batting_model_path(self) -> str:
        """打撃モデルファイルのパスを取得"""
        return self.get_models_path(FilePaths.BATTING_MODEL)
    
    def get_nn_model_path(self) -> str:
        """ニューラルネットワークモデルファイルのパスを取得"""
        return self.get_models_path(FilePaths.NN_MODEL)
    
    def ensure_directory_exists(self, path: str) -> bool:
        """ディレクトリが存在することを確認し、なければ作成"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False
    
    def file_exists(self, path: str) -> bool:
        """ファイルが存在するかチェック"""
        return Path(path).is_file()
    
    @staticmethod
    def ensure_directory_exists_static(path: str) -> bool:
        """ディレクトリが存在することを確認し、なければ作成（静的メソッド）"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False


# シングルトンインスタンス
path_manager = PathManager()
