"""
共通ユーティリティモジュール
プロジェクト全体で使用される汎用的な関数やクラスを提供
"""
import os
import json
from typing import Any, Dict, Optional
from pathlib import Path


class FileUtils:
    """ファイル操作のユーティリティクラス"""
    
    @staticmethod
    def ensure_directory_exists(path: str) -> None:
        """ディレクトリが存在しない場合は作成"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
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
            FileUtils.ensure_directory_exists(os.path.dirname(filepath))
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, TypeError) as e:
            print(f"Error saving JSON to {filepath}: {e}")
            return False


class ValidationUtils:
    """バリデーション関連のユーティリティクラス"""
    
    @staticmethod
    def validate_percentage(value: float, name: str = "value") -> bool:
        """パーセンテージ値（0-100）の妥当性チェック"""
        if not isinstance(value, (int, float)):
            print(f"Error: {name} must be a number")
            return False
        if not 0 <= value <= 100:
            print(f"Error: {name} must be between 0 and 100")
            return False
        return True
    
    @staticmethod
    def validate_required_keys(data: Dict, required_keys: list, context: str = "data") -> bool:
        """必須キーの存在チェック"""
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"Error: Missing required keys in {context}: {missing_keys}")
            return False
        return True


class StringUtils:
    """文字列処理のユーティリティクラス"""
    
    @staticmethod
    def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
        """文字列を指定された長さで切り詰める"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """パーセンテージ値を整形"""
        return f"{value:.{decimal_places}f}%"
    
    @staticmethod
    def safe_strip(text: Optional[str]) -> str:
        """安全な文字列のstrip（Noneでもエラーにならない）"""
        return text.strip() if text else ""


class MathUtils:
    """数学関連のユーティリティクラス"""
    
    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """値を指定された範囲内にクランプ"""
        return max(min_value, min(max_value, value))
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全な除算（ゼロ除算を回避）"""
        return numerator / denominator if denominator != 0 else default
    
    @staticmethod
    def round_to_decimal_places(value: float, places: int) -> float:
        """指定された小数点以下桁数で四捨五入"""
        return round(value, places)
