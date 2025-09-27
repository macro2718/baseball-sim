"""Fetch 2024 Dodgers hitters (>=50 PA) and print Name, Pos (from statsapi), K%, BB%, Hard%, GB%, PA.

Optionally fetch primaryPosition using MLB StatsAPI (statsapi). If statsapi lookup fails,
Pos will be blank for that player. Network calls are cached minimally in-memory per run.

Usage (after installing requirements including pybaseball):
    python -m scripts.dodgers_2024_batting
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Dict

import pandas as pd

try:
    from pybaseball import batting_stats
except ImportError as e:  # pragma: no cover - runtime guidance
    raise SystemExit("pybaseball not installed. Add it to requirements and pip install.") from e

try:
    import statsapi  # type: ignore
except ImportError:
    statsapi = None  # graceful degradation


def fetch_dodgers_2024(min_pa: int = 50, debug: bool = False, include_position: bool = True) -> pd.DataFrame:
    """Return DataFrame of Dodgers hitters 2024 with requested rate stats.

    Parameters
    ----------
    min_pa: int
        Minimum plate appearances to include.
    """
    # Suppress any minor data warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = batting_stats(2024, qual=min_pa)

    if debug:
        print("DEBUG batting columns:", df.columns.tolist())

    # Filter to Dodgers (team abbreviation 'LAD')
    lad = df[df["Team"] == "LAD"].copy()

    # Resolve player name column (pybaseball version differences)
    name_col_candidates = ["Name", "Player", "name"]
    name_col: Optional[str] = None
    for cand in name_col_candidates:
        if cand in lad.columns:
            name_col = cand
            break
    if name_col is None:
        raise KeyError(f"Could not find a player name column among candidates {name_col_candidates}. Columns present: {lad.columns.tolist()}")

    # Some columns we want may have different canonical names depending on pybaseball version.
    # We'll map possible variants.
    col_map_variants = {
        "K%": ["K%", "SO%"],
        "BB%": ["BB%"],
        "Hard%": ["Hard%", "HardHit%", "HardHit Rate"],
        "GB%": ["GB%", "GB% (Balls In Play)"]
    }

    def resolve(col_alternatives: List[str]) -> str:
        for c in col_alternatives:
            if c in lad.columns:
                return c
        raise KeyError(f"None of {col_alternatives} present in batting stats columns: {lad.columns.tolist()}")

    resolved = {k: resolve(v) for k, v in col_map_variants.items()}

    # Prepare output columns (without position first)
    out_cols = [name_col, resolved["K%"], resolved["BB%"], resolved["Hard%"], resolved["GB%"], "PA"]
    present_cols = [c for c in out_cols if c in lad.columns]

    result = lad[present_cols].sort_values("PA", ascending=False).reset_index(drop=True)

    # Rename resolved columns to requested canonical names
    rename_map = {resolved["K%"]: "K%", resolved["BB%"]: "BB%", resolved["Hard%"]: "Hard%", resolved["GB%"]: "GB%", name_col: "Name"}
    result = result.rename(columns=rename_map)

    # Optionally enrich position using statsapi (primaryPosition)
    if include_position and statsapi is not None:
        pos_cache: Dict[str, str] = {}

        def fetch_primary_position(player_name: str) -> str:
            if player_name in pos_cache:
                return pos_cache[player_name]
            try:
                search = statsapi.lookup_player(player_name)
                if not search:
                    pos_cache[player_name] = ""
                    return ""
                # Heuristic: choose first exact (case-insensitive) name match, else first.
                lower = player_name.lower()
                selected = None
                for cand in search:
                    if cand.get('fullName','').lower() == lower:
                        selected = cand
                        break
                if selected is None:
                    selected = search[0]
                pos_info = selected.get('primaryPosition', {}) or {}
                abbrev = pos_info.get('abbreviation') or pos_info.get('code') or ""
                pos_cache[player_name] = abbrev
                return abbrev
            except Exception:
                pos_cache[player_name] = ""
                return ""

        result['Pos'] = result['Name'].apply(fetch_primary_position)
        # Reorder columns to place Pos after Name
        ordered_cols = [c for c in ["Name", "Pos", "K%", "BB%", "Hard%", "GB%", "PA"] if c in result.columns]
        result = result[ordered_cols]
    else:
        if include_position and statsapi is None and debug:
            print("DEBUG: statsapi not installed; skipping position enrichment")
        result = result[[c for c in ["Name", "K%", "BB%", "Hard%", "GB%", "PA"] if c in result.columns]]

    return result


def main():  # pragma: no cover
    df = fetch_dodgers_2024()
    if df.empty:
        print("No data returned.")
        return
    # Pretty print
    print(df.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
