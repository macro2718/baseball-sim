"""Fetch 2024 Dodgers hitters (>=50 PA) and print Name, Pos (primary), optional position splits (>=10G), K%, BB%, Hard%, GB%, PA.

Optionally fetch primaryPosition using MLB StatsAPI (statsapi). If statsapi lookup fails,
Pos will be blank for that player. Additionally, when enabled, fetch all defensive
positions at which the player has appeared in a minimum number of games (default 10)
for the season, producing a column like ``Pos(>=10G)``.

Network calls are cached minimally in-memory per run.

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


def fetch_dodgers_2024(
    min_pa: int = 50,
    debug: bool = False,
    include_position: bool = True,
    *,
    season: int = 2024,
    include_position_splits: bool = True,
    min_position_games: int = 10,
) -> pd.DataFrame:
    """Return DataFrame of Dodgers hitters for a season with requested rate stats.

    Parameters
    ----------
    min_pa: int
        Minimum plate appearances to include.
    debug: bool
        If True, print helpful debug information.
    include_position: bool
        If True, include primary position ("Pos") via statsapi lookup.
    season: int
        MLB season year (defaults to 2024).
    include_position_splits: bool
        If True and statsapi available, also derive an additional column listing
        all defensive positions where the player has appeared in at least
        ``min_position_games`` games for the given season (e.g. "C,1B,LF").
        This will appear in a column named "Pos(>=10G)" by default.
    min_position_games: int
        Minimum number of games at a position for it to be included in the
        position splits column.
    """
    # Suppress any minor data warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    df = batting_stats(season, qual=min_pa)

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
        id_cache: Dict[str, int] = {}

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
                # cache numeric player id if present
                pid = selected.get('id') or selected.get('player_id')
                if isinstance(pid, int):
                    id_cache[player_name] = pid
                pos_info = selected.get('primaryPosition', {}) or {}
                abbrev = pos_info.get('abbreviation') or pos_info.get('code') or ""
                pos_cache[player_name] = abbrev
                return abbrev
            except Exception:
                pos_cache[player_name] = ""
                return ""

        result['Pos'] = result['Name'].apply(fetch_primary_position)

        # Optionally fetch position split appearances (>= min_position_games)
        if include_position_splits:
            pos_splits_cache: Dict[str, str] = {}

            def fetch_position_splits(player_name: str) -> str:
                if player_name in pos_splits_cache:
                    return pos_splits_cache[player_name]
                pid = id_cache.get(player_name)
                if pid is None:
                    # Attempt another lookup if id was missing first time
                    try:
                        search = statsapi.lookup_player(player_name)
                        if search:
                            pid_candidate = search[0].get('id') or search[0].get('player_id')
                            if isinstance(pid_candidate, int):
                                pid = pid_candidate
                                id_cache[player_name] = pid_candidate
                    except Exception:
                        if debug:
                            print(f"DEBUG: secondary lookup failed for {player_name}")
                if pid is None:
                    if debug:
                        print(f"DEBUG: no player id for splits: {player_name}")
                    pos_splits_cache[player_name] = ""
                    return ""
                try:
                    # statsapi.get("person", {"personId": pid, "hydrate": "stats(group=[\"fielding\"],type=[\"season\"],season=2024)"}) pattern
                    hydrate = f"stats(group=[fielding],type=[season],season={season})"
                    data = statsapi.get("person", {"personId": pid, "hydrate": hydrate})  # type: ignore
                    stats_list = (((data or {}).get('people') or [{}])[0].get('stats')) or []
                    positions: Dict[str, int] = {}
                    for entry in stats_list:
                        if (entry or {}).get('group', {}).get('displayName') != 'fielding':
                            continue
                        splits = entry.get('splits') or []
                        for s in splits:
                            stat = s.get('stat') or {}
                            games = stat.get('gamesPlayed') or stat.get('games')
                            pos = (s.get('position') or {}).get('abbreviation') or (s.get('position') or {}).get('code')
                            if not pos or games is None:
                                continue
                            try:
                                g_int = int(games)
                            except (TypeError, ValueError):
                                continue
                            positions[pos] = positions.get(pos, 0) + g_int
                    qualified = [p for p, g in sorted(positions.items(), key=lambda x: (-x[1], x[0])) if g >= min_position_games]
                    value = ",".join(qualified)
                    pos_splits_cache[player_name] = value
                    return value
                except Exception:
                    if debug:
                        print(f"DEBUG: splits fetch failed for {player_name}")
                    pos_splits_cache[player_name] = ""
                    return ""

            result['Pos(>=' + str(min_position_games) + 'G)'] = result['Name'].apply(fetch_position_splits)

        # Reorder columns to place Pos and Pos(>=xG) after Name
        base_order = ["Name", "Pos"]
        if include_position_splits:
            base_order.append('Pos(>=' + str(min_position_games) + 'G)')
        base_order += ["K%", "BB%", "Hard%", "GB%", "PA"]
        ordered_cols = [c for c in base_order if c in result.columns]
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
