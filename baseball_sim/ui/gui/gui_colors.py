"""
GUI color helpers for baseball position tokens.
"""
from typing import Optional


def get_position_color(position: Optional[str], pitcher_type: Optional[str] = None) -> Optional[str]:
    """Return a foreground color name for a given position.

    Rules (as requested):
    - SP (先発ピッチャー): red
    - RP (リリーフピッチャー): purple
    - C (キャッチャー): blue
    - 1B, 2B, 3B, SS (内野): yellow
    - LF, CF, RF (外野): green

    If position is 'P', decide using pitcher_type.
    Returns None if no specific color mapping applies.
    """
    if not position:
        return None

    pos = position.upper()

    # Direct pitcher role tokens
    if pos == "SP":
        return "red"
    if pos == "RP":
        return "purple"

    # Catcher
    if pos == "C":
        return "blue"

    # Infield
    if pos in {"1B", "2B", "3B", "SS"}:
        return "orange"

    # Outfield
    if pos in {"LF", "CF", "RF"}:
        return "green"

    # Generic pitcher position -> rely on pitcher_type
    if pos == "P":
        if pitcher_type and pitcher_type.upper() == "SP":
            return "red"
        if pitcher_type and pitcher_type.upper() == "RP":
            return "purple"
        # Default for pitchers if type unknown
        return "red"

    return None

