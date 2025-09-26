"""統計計算処理を統一するクラス"""


class StatsCalculator:
    """統計計算の統一クラス"""
    
    @staticmethod
    def calculate_batting_average(hits, at_bats):
        """打率計算（重複していた処理を統一）"""
        return hits / at_bats if at_bats > 0 else 0.000
    
    @staticmethod
    def _outs_from_innings(innings_pitched):
        """Convert an innings value into the number of recorded outs.

        The simulation stores投球回 as累積した1/3回 increments, which leads to
        floating point rounding issues (e.g. 1.9999999 instead of exactly 2.0).
        Additionally、外部データが"1.2"のような表記で提供される場合にも対応する。
        この補助関数では入力値をアウト数へ正規化し、以降の集計処理で一貫して
        利用できるようにする。
        """

        if innings_pitched is None:
            return 0

        # 文字列表記 (例: "1.2") にも対応する
        if isinstance(innings_pitched, str):
            innings_str = innings_pitched.strip()
        else:
            innings_str = str(innings_pitched)

        if "." in innings_str:
            whole_str, frac_str = innings_str.split(".", 1)
            if frac_str in {"", "0", "1", "2"}:
                try:
                    whole = int(whole_str) if whole_str else 0
                    frac_outs = int(frac_str or "0")
                except ValueError:
                    pass
                else:
                    if whole >= 0 and 0 <= frac_outs <= 2:
                        return whole * 3 + frac_outs

        try:
            numeric_value = float(innings_pitched)
        except (TypeError, ValueError):
            return 0

        outs = int(round(numeric_value * 3))
        return max(outs, 0)

    @staticmethod
    def _normalized_innings(innings_pitched):
        """正規化した投球回（浮動小数）を返す。"""

        outs = StatsCalculator._outs_from_innings(innings_pitched)
        return outs / 3 if outs > 0 else 0.0

    @staticmethod
    def calculate_era(earned_runs, innings_pitched):
        """防御率計算（重複していた処理を統一）"""
        innings = StatsCalculator._normalized_innings(innings_pitched)
        return (earned_runs * 9) / innings if innings > 0 else 0.00
    
    @staticmethod
    def calculate_whip(hits, walks, innings_pitched):
        """WHIP計算（重複していた処理を統一）"""
        innings = StatsCalculator._normalized_innings(innings_pitched)
        return (hits + walks) / innings if innings > 0 else 0.00
    
    @staticmethod
    def calculate_obp(hits, walks, at_bats, hit_by_pitch=0, sacrifice_flies=0):
        """出塁率計算（統一処理）"""
        plate_appearances = at_bats + walks + hit_by_pitch + sacrifice_flies
        return (hits + walks + hit_by_pitch) / plate_appearances if plate_appearances > 0 else 0.000
    
    @staticmethod
    def calculate_slg(singles, doubles, triples, home_runs, at_bats):
        """長打率計算（統一処理）"""
        total_bases = singles + (doubles * 2) + (triples * 3) + (home_runs * 4)
        return total_bases / at_bats if at_bats > 0 else 0.000
    
    @staticmethod
    def calculate_ops(obp, slg):
        """OPS計算（統一処理）"""
        return obp + slg
    
    @staticmethod
    # Removed unused calculate_babip
    
    @staticmethod
    def calculate_k_per_9(strikeouts, innings_pitched):
        """K/9計算（統一処理）"""
        innings = StatsCalculator._normalized_innings(innings_pitched)
        return (strikeouts * 9) / innings if innings > 0 else 0.00
    
    @staticmethod
    def calculate_bb_per_9(walks, innings_pitched):
        """BB/9計算（統一処理）"""
        innings = StatsCalculator._normalized_innings(innings_pitched)
        return (walks * 9) / innings if innings > 0 else 0.00
    
    @staticmethod
    def calculate_hr_per_9(home_runs, innings_pitched):
        """HR/9計算（統一処理）"""
        innings = StatsCalculator._normalized_innings(innings_pitched)
        return (home_runs * 9) / innings if innings > 0 else 0.00

    @staticmethod
    def record_strikeout(stats):
        """記録上の三振数を更新し、SOと旧来のKキーを同期させる"""
        current = stats.get("SO", stats.get("K", 0)) + 1
        stats["SO"] = current
        stats["K"] = current
        return current

    @staticmethod
    def format_inning_display(innings_pitched):
        """投球イニングの表示形式を統一"""
        outs = StatsCalculator._outs_from_innings(innings_pitched)
        innings = outs // 3
        remainder = outs % 3
        return f"{innings}.{remainder}"
    
    @staticmethod
    def format_average(value, decimal_places=3):
        """打率等の数値表示形式を統一"""
        if decimal_places == 3:
            return f".{int(value*1000):03d}"
        elif decimal_places == 2:
            return f"{value:.2f}"
        else:
            return f"{value:.{decimal_places}f}"
