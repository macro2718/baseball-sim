"""
共通ユーティリティモジュール
プロジェクト全体で使用される汎用的な関数やクラスを提供
（ファイル操作はpath_utils.pyに移動済み）
"""
from typing import Any, Dict, List


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
    def validate_required_keys(data: Dict, required_keys: List[str], context: str = "data") -> bool:
        """必須キーの存在チェック"""
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"Error: Missing required keys in {context}: {missing_keys}")
            return False
        return True
    
    @staticmethod
    def validate_positive_number(value: Any, name: str = "value") -> bool:
        """正の数値の妥当性チェック"""
        if not isinstance(value, (int, float)):
            print(f"Error: {name} must be a number")
            return False
        if value <= 0:
            print(f"Error: {name} must be positive")
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
    def safe_string_conversion(value: Any) -> str:
        """安全な文字列変換"""
        if value is None:
            return ""
        return str(value)


class MathUtils:
    """数学関連のユーティリティクラス"""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """値を指定された範囲内に制限"""
        return max(min_val, min(value, max_val))
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全な除算（ゼロ除算エラーを防ぐ）"""
        if denominator == 0:
            return default
        return numerator / denominator
    
    @staticmethod
    def round_to_places(value: float, places: int = 2) -> float:
        """指定された小数点以下の桁数で四捨五入"""
        return round(value, places)
    
    @staticmethod
    def round_to_decimal_places(value: float, places: int) -> float:
        """指定された小数点以下桁数で四捨五入"""
        return round(value, places)
