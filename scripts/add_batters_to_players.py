"""Append MLB batters for a given season (>= MIN_PA PA) to players.json with adjusted naming and defensively eligible positions.

Updated Position Rules (2025 spec):
 1. For each player we fetch fielding splits via statsapi (if available) using a season fielding hydrate.
 2. Count games played at each defensive position.
 3. If the player has one or more positions with Games >= POSITION_MIN_G (default 10), all such positions become the defensive eligible positions.
 4. If no position reaches the threshold, choose only the single position with the highest games (ties broken alphabetically).
 5. Always append 'DH' to the eligible positions list (ensuring at least ['DH']).
 6. If statsapi is unavailable or the lookup fails, fall back to ['DH'] only.

Bat Side Determination:
 - Each player's batting side (bats) is now pulled from statsapi person lookup (`batSide.code` -> 'L', 'R', or 'S').
 - If unavailable (statsapi off or lookup failure) defaults to 'R'.
 - Backfill will also update bats if previously defaulted.

Other Behavior:
 - Fetch YEAR batting stats (qual=MIN_PA) via pybaseball.
 - Avoid duplicates by case-insensitive comparison of base name (ignoring seasonal suffix).
 - Name suffix logic unchanged (e.g. "('24)"), configurable via SUFFIX.
 - Backfill mode updates existing entries for the SAME seasonal suffix that currently only have ['DH'] -> recalculated defensive set + 'DH'.
 - Generate id as UUID v4.

Usage:
    python -m scripts.add_2024_batters_to_players

Configuration Notes:
    YEAR, SUFFIX, PLAYERS_FILE, MIN_PA, DRY_RUN, LIMIT, USE_STATSAPI, BACKFILL_POSITIONS, TEAM as documented inline below.
    POSITION_MIN_G sets the >= games threshold (default 10).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

try:
    from pybaseball import batting_stats
except ImportError as e:
    raise SystemExit("pybaseball not installed; install dependencies first.") from e

try:
    import statsapi  # type: ignore
except ImportError:
    statsapi = None


POSITION_MIN_G = 10  # Threshold for multi-position eligibility

# =============================
# Configuration (edit as needed)
# =============================
YEAR = 2024  # Season year to pull
# If you want automatic suffix "('YY)" leave SUFFIX = None. Otherwise set explicit e.g. "(2024)" or "_2024".
SUFFIX: str | None = None
PLAYERS_FILE = 'player_data/data/players.json'
MIN_PA = 50              # Minimum plate appearances to include
DRY_RUN = False          # True = don't write file
LIMIT = 0                # 0 = no limit; otherwise process first N new players
USE_STATSAPI = True      # False: skip statsapi fielding splits (all players => ['DH'])
BACKFILL_POSITIONS = False  # True to backfill defensive positions for existing DH-only entries for this YEAR
TEAM = 'COL'                # e.g. 'LAD' to restrict to one team; '' for all teams

def _computed_suffix() -> str:
    if SUFFIX is not None and SUFFIX != '':
        return SUFFIX
    yy = str(YEAR)[-2:]
    return f"('{yy})"

NAME_SUFFIX = _computed_suffix()

# Folder label combining TEAM and season suffix, e.g. TEAM='LAD' and NAME_SUFFIX="('24)" -> "LAD('24)"
FOLDER_LABEL = f"{TEAM}{NAME_SUFFIX}" if TEAM and NAME_SUFFIX else (TEAM or NAME_SUFFIX or '')


def normalize_existing_name(name: str) -> str:
    """Remove the current NAME_SUFFIX (seasonal suffix) if present and lower-case.
    Used to compare existing players ignoring the seasonal suffix."""
    if NAME_SUFFIX and name.endswith(NAME_SUFFIX):
        base = name[: -len(NAME_SUFFIX)]
    else:
        base = name
    return base.strip().lower()


def build_batter_entry(row: Dict, pos: str, bat_side: str) -> Dict:
    name = row['Name']
    suffixed = name if (NAME_SUFFIX and name.endswith(NAME_SUFFIX)) else f"{name}{NAME_SUFFIX}" if NAME_SUFFIX else name
    eligible: List[str] = []
    if pos:
        eligible.append(pos)
    if 'DH' not in eligible:
        eligible.append('DH')
    return {
        'name': suffixed,
        'k_pct': round(float(row['K%'])*100, 2),
        'bb_pct': round(float(row['BB%'])*100, 2),
        'hard_pct': round(float(row['Hard%'])*100, 2),
        'gb_pct': round(float(row['GB%'])*100, 2),
        'eligible_positions': eligible if eligible else ['DH'],
        'bats': bat_side if bat_side in {'L','R','S'} else 'R',
        'speed': 100.0,
        'fielding_skill': 100.0,
        'id': str(uuid.uuid4())
    }


def main() -> int:
    players_path = Path(PLAYERS_FILE)
    if not players_path.exists():
        raise SystemExit(f"Players file not found: {players_path}")

    with players_path.open('r', encoding='utf-8') as f:
        players_data = json.load(f)

    # Ensure top-level folders list contains our team-season folder label if defined
    mutated_top_level = False
    if FOLDER_LABEL:
        folders_list = players_data.setdefault('folders', [])
        if isinstance(folders_list, list) and FOLDER_LABEL not in folders_list:
            folders_list.append(FOLDER_LABEL)
            mutated_top_level = True

    existing_batters = players_data.get('batters', [])
    existing_keys: Set[str] = {normalize_existing_name(b['name']) for b in existing_batters}

    # Cache for bat side codes: normalized player base name -> 'L'/'R'/'S'
    bat_side_cache: Dict[str, str] = {}

    # Helper: fetch fielding games by position
    def _fetch_fielding_positions(player_name: str, debug: bool = False) -> Dict[str, int]:
        if statsapi is None:
            return {}
        try:
            # First get lookup to resolve id
            res = statsapi.lookup_player(player_name)
            if not res:
                return {}
            lower = player_name.lower()
            chosen = None
            for r in res:
                if r.get('fullName', '').lower() == lower:
                    chosen = r
                    break
            if chosen is None:
                chosen = res[0]
            # capture bat side if present; fallback to person endpoint if missing
            bat_side = (chosen.get('batSide') or {}).get('code') if isinstance(chosen, dict) else None
            if not bat_side:
                try:
                    # minimal person fetch (without heavy hydrate) to get batSide
                    person_data = statsapi.get('person', {"personId": chosen.get('id') or chosen.get('player_id')})  # type: ignore
                    people = (person_data or {}).get('people') or []
                    if people:
                        bat_side = (people[0].get('batSide') or {}).get('code')
                except Exception:
                    if debug:
                        print(f"DEBUG batSide fallback failed for {player_name}")
            if isinstance(bat_side, str) and bat_side:
                bat_side_cache[normalize_existing_name(player_name)] = bat_side.upper()[0]
            elif debug:
                print(f"DEBUG batSide missing; defaulting to 'R' for {player_name}")
            pid = chosen.get('id') or chosen.get('player_id')
            if not pid:
                return {}
            hydrate = f"stats(group=[fielding],type=[season],season={YEAR})"
            data = statsapi.get('person', {"personId": pid, "hydrate": hydrate})  # type: ignore
            stats_list = (((data or {}).get('people') or [{}])[0].get('stats')) or []
            pos_games: Dict[str, int] = {}
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
                    # Directly use reported position code; no TWP normalization needed here
                    pos_games[pos] = pos_games.get(pos, 0) + g_int
            return pos_games
        except Exception:
            return {}

    def derive_eligible_positions(player_name: str) -> List[str]:
        pos_games = _fetch_fielding_positions(player_name, debug=False)
        if not pos_games:
            return ['DH']
        # Determine threshold positions
        threshold = [p for p, g in pos_games.items() if g >= POSITION_MIN_G]
        if threshold:
            chosen = sorted(threshold, key=lambda x: ( -pos_games[x], x))
        else:
            # pick single max (alphabetical tie)
            max_g = max(pos_games.values())
            candidates = [p for p, g in pos_games.items() if g == max_g]
            chosen = [sorted(candidates)[0]]
        if 'DH' not in chosen:
            chosen.append('DH')
        return chosen

    if BACKFILL_POSITIONS:
        if DRY_RUN:
            mode_msg = 'Backfill (dry-run)'
        else:
            mode_msg = 'Backfill'
        updated = 0
        for b in existing_batters:
            name = b.get('name')
            if not name or (NAME_SUFFIX and not name.endswith(NAME_SUFFIX)):
                continue
            elig = b.get('eligible_positions', [])
            # Remove legacy 'Bats' column fallback: bats now determined via statsapi batSide.
            non_dh = [p for p in elig if p != 'DH']
            if non_dh:
                continue  # already has defensive spot
            # reconstruct display base (remove suffix)
            display_name = name[:-len(NAME_SUFFIX)] if NAME_SUFFIX and name.endswith(NAME_SUFFIX) else name
            new_positions: List[str] = []
            if USE_STATSAPI:
                new_positions = derive_eligible_positions(display_name)
            if not new_positions:
                new_positions = ['DH']
            b['eligible_positions'] = new_positions
            updated += 1
            print(f"Backfilled {name}: positions={new_positions}")
        if updated == 0:
            print(f"{mode_msg}: no entries needed updating.")
        else:
            print(f"{mode_msg}: updated {updated} entries.")
            if not DRY_RUN:
                with players_path.open('w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
                print(f"Wrote back updates to {players_path}")
        # If no positional updates but we inserted a new folder tag, write it
        if mutated_top_level and not DRY_RUN:
            with players_path.open('w', encoding='utf-8') as f:
                json.dump(players_data, f, ensure_ascii=False, indent=2)
            print(f"Added folder '{FOLDER_LABEL}' to top-level folders.")
        return 0

    df = batting_stats(YEAR, qual=MIN_PA)
    # Optional team filter
    if TEAM:
        if 'Team' not in df.columns:
            print(f"Warning: TEAM specified ({TEAM}) but 'Team' column not present in data; ignoring filter.")
        else:
            before = len(df)
            df = df[df['Team'] == TEAM].copy()
            print(f"Filtered to team {TEAM}: {len(df)} rows (from {before}).")
    # Batting side comes exclusively from statsapi (not pybaseball columns).

    # Build candidate rows
    candidates: List[Dict] = []
    for _, row in df.iterrows():
        name = row['Name'] if 'Name' in row else row.get('Player')
        if not name:
            continue
        base_name_key = normalize_existing_name(name)
        if base_name_key in existing_keys:
            continue  # skip duplicates
        positions: List[str] = []
        if USE_STATSAPI:
            positions = derive_eligible_positions(name)
        if not positions:
            positions = ['DH']
        primary_for_entry = next((p for p in positions if p != 'DH'), '')
        bat_side = bat_side_cache.get(base_name_key, 'R')
        entry = build_batter_entry(row, primary_for_entry, bat_side)
        # Attach folder tag to new player
        if FOLDER_LABEL:
            entry['folders'] = [FOLDER_LABEL]
        entry['eligible_positions'] = positions
        candidates.append(entry)

    if LIMIT > 0:
        candidates = candidates[:LIMIT]

    if not candidates:
        if mutated_top_level:
            if DRY_RUN:
                print(f"Dry run: would have added folder '{FOLDER_LABEL}' to top-level folders (no new players).")
            else:
                with players_path.open('w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
                print(f"Added folder '{FOLDER_LABEL}' (no new players to add).")
        else:
            print('No new players to add.')
        return 0
    for c in candidates:
        print(f" - {c['name']} Pos={c['eligible_positions'][0]} K%={c['k_pct']} BB%={c['bb_pct']}")

    if DRY_RUN:
        print('\nDry run enabled; no file written.')
        return 0

    # Append and write back
    players_data.setdefault('batters', []).extend(candidates)
    # Keep JSON pretty & stable ordering
    with players_path.open('w', encoding='utf-8') as f:
        json.dump(players_data, f, ensure_ascii=False, indent=2)
    if mutated_top_level:
        # Ensure mutated_top_level persists in saved file (folders list already mutated in memory)
        pass
    print(f"Updated {players_path} with {len(candidates)} new batters (folder tag='{FOLDER_LABEL}').")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
