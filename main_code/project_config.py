"""
プロジェクト共通設定モジュール
プロジェクトのパス管理とインポート設定を一元化
"""
import os
import sys
from pathlib import Path


class ProjectPaths:
    """プロジェクトパス管理クラス"""
    
    def __init__(self):
        # main_codeディレクトリの絶対パス
        self._main_code_dir = Path(__file__).parent.absolute()
        # プロジェクトルートディレクトリ（main_codeの親ディレクトリ）
        self._project_root = self._main_code_dir.parent
    
    @property
    def project_root(self) -> Path:
        """プロジェクトルートディレクトリ"""
        return self._project_root
    
    @property
    def main_code_dir(self) -> Path:
        """main_codeディレクトリ"""
        return self._main_code_dir
    
    @property
    def player_data_dir(self) -> Path:
        """player_dataディレクトリ"""
        return self._project_root / "player_data"
    
    @property
    def prediction_models_dir(self) -> Path:
        """prediction_modelsディレクトリ"""
        return self._project_root / "prediction_models"
    
    @property
    def data_dir(self) -> Path:
        """データファイルディレクトリ"""
        return self.player_data_dir / "data"
    
    @property
    def models_dir(self) -> Path:
        """モデルファイルディレクトリ"""
        return self.prediction_models_dir / "models"
    
    def ensure_project_paths_in_sys_path(self):
        """必要なパスをsys.pathに追加（重複チェック付き）"""
        paths_to_add = [
            str(self.project_root),
            str(self.main_code_dir),
            str(self.player_data_dir),
            str(self.prediction_models_dir)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)


# グローバルインスタンス
_project_paths = ProjectPaths()

def setup_project_environment():
    """プロジェクト環境をセットアップ"""
    _project_paths.ensure_project_paths_in_sys_path()

def get_project_paths() -> ProjectPaths:
    """プロジェクトパス管理インスタンスを取得"""
    return _project_paths

# 自動的に環境をセットアップ
setup_project_environment()
