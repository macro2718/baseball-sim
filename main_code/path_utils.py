"""
ファイルパスユーティリティ
プロジェクト内のファイルパス管理を一元化
"""
import os
from typing import Union
from constants import FilePaths

class PathManager:
    """ファイルパス管理クラス"""
    
    def __init__(self, base_dir: str = None):
        # プロジェクトルートディレクトリを基準にする
        if base_dir is None:
            # main_codeから一つ上のディレクトリ（プロジェクトルート）を取得
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
    
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
            os.makedirs(path, exist_ok=True)
            return True
        except OSError:
            return False
    
    def file_exists(self, path: str) -> bool:
        """ファイルが存在するかチェック"""
        return os.path.isfile(path)

# シングルトンインスタンス
path_manager = PathManager()
