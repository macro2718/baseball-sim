"""Append MLB pitchers for a given season (>= MIN_IP innings pitched) to players.json with team-season folder tagging.

Features:
 - Pull season pitching stats via pybaseball (pitching_stats) with a minimum innings threshold.
 - Determine pitcher_type (SP vs RP) using Games Started (GS) vs total Games (G).
   * Heuristic: if GS / G >= STARTER_THRESHOLD (default 0.5) and GS >= MIN_GS_FOR_SP (default 5) -> SP else RP.
 - Retrieve throwing hand from statsapi person lookup (pitchHand.code) -> 'L' or 'R'. Defaults to 'R' if unavailable.
 - Assign stamina based on role (configurable constants) or simple scaling by IP.
 - Avoid duplicates by case-insensitive comparison of existing pitcher names ignoring the seasonal suffix.
 - Add folder tag (e.g. LAD('24)) to each new pitcher and ensure top-level folders list contains it.
 - Backfill mode (optional) could be added later (not implemented now to keep scope minimal).

Usage:
    python -m scripts.add_pitchers_to_players

Configuration:
    YEAR, SUFFIX, PLAYERS_FILE, MIN_IP (reliever min), MIN_IP_SP (starter min), DRY_RUN, LIMIT,
    TEAM, STARTER_THRESHOLD, MIN_GS_FOR_SP, STAMINA_SP, STAMINA_RP, USE_STATSAPI.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

try:
    from pybaseball import pitching_stats
except ImportError as e:
    raise SystemExit("pybaseball not installed; install dependencies first.") from e

try:
    import statsapi  # type: ignore
except ImportError:
    statsapi = None

# =============================
# Configuration
# =============================
YEAR = 2024
SUFFIX: str | None = None  # if None -> auto ("('YY)")
PLAYERS_FILE = 'player_data/data/players.json'
MIN_IP = 20          # Minimum innings pitched for relievers (query + RP filter)
MIN_IP_SP = 50       # Minimum innings pitched for starters
DRY_RUN = False       # True -> do not write file
LIMIT = 0             # 0 = no limit
TEAM = 'COL'          # '' for all teams, else team code
USE_STATSAPI = True
STARTER_THRESHOLD = 0.5  # proportion of games that are starts to be SP
MIN_GS_FOR_SP = 5        # minimum games started requirement for SP
STAMINA_SP = 80          # default stamina for starters
STAMINA_RP = 40          # default stamina for relievers


def _computed_suffix() -> str:
    if SUFFIX is not None and SUFFIX != '':
        return SUFFIX
    return f"('{str(YEAR)[-2:]})"

NAME_SUFFIX = _computed_suffix()
FOLDER_LABEL = f"{TEAM}{NAME_SUFFIX}" if TEAM and NAME_SUFFIX else (TEAM or NAME_SUFFIX or '')


def normalize_existing_name(name: str) -> str:
    if NAME_SUFFIX and name.endswith(NAME_SUFFIX):
        base = name[:-len(NAME_SUFFIX)]
    else:
        base = name
    return base.strip().lower()


def classify_pitcher(row: Dict) -> str:
    gs = row.get('GS') or 0
    g = row.get('G') or 0
    try:
        gs = float(gs)
        g = float(g)
    except Exception:
        return 'RP'
    if g <= 0:
        return 'RP'
    if gs >= MIN_GS_FOR_SP and (gs / g) >= STARTER_THRESHOLD:
        return 'SP'
    return 'RP'


def compute_stamina(role: str, innings: float) -> int:
    # Simple heuristic: base by role; slight scaling with innings (capped)
    if role == 'SP':
        base = STAMINA_SP
    else:
        base = STAMINA_RP
    # scale: + (IP / 100)*10 but cap at base+20
    bonus = min(20, int((innings / 100.0) * 10))
    return int(base + bonus)


def build_pitcher_entry(row: Dict, pitcher_type: str, throws: str, stamina: int) -> Dict:
    name = row['Name']
    suffixed = name if (NAME_SUFFIX and name.endswith(NAME_SUFFIX)) else f"{name}{NAME_SUFFIX}" if NAME_SUFFIX else name
    # K% and BB% may not be directly present; pybaseball pitching_stats includes K% and BB% columns (already % values)
    k_pct_val = row.get('K%')
    bb_pct_val = row.get('BB%')
    hard_pct_val = row.get('Hard%')
    gb_pct_val = row.get('GB%')
    # Some columns might be missing; fallback to 0
    def pct(x):
        try:
            return float(x) if x is not None else 0.0
        except Exception:
            return 0.0
    return {
        'name': suffixed,
        'pitcher_type': pitcher_type,
        'k_pct': round(pct(k_pct_val)*100, 2),
        'bb_pct': round(pct(bb_pct_val)*100, 2),
        'hard_pct': round(pct(hard_pct_val)*100, 2),
        'gb_pct': round(pct(gb_pct_val)*100, 2),
        'stamina': stamina,
        'throws': 'L' if throws == 'L' else 'R',
        'id': str(uuid.uuid4())
    }


# Cache for throwing hand; normalized name -> 'L'/'R'
throws_cache: Dict[str, str] = {}


def fetch_throws(player_name: str) -> str:
    if not USE_STATSAPI or statsapi is None:
        return 'R'
    key = normalize_existing_name(player_name)
    if key in throws_cache:
        return throws_cache[key]
    try:
        res = statsapi.lookup_player(player_name)
        if not res:
            return 'R'
        lower = player_name.lower()
        chosen = None
        for r in res:
            if r.get('fullName', '').lower() == lower:
                chosen = r
                break
        if chosen is None:
            chosen = res[0]
        throws = (chosen.get('pitchHand') or {}).get('code')
        if not throws:
            # fallback direct person fetch
            try:
                person_data = statsapi.get('person', {"personId": chosen.get('id') or chosen.get('player_id')})  # type: ignore
                people = (person_data or {}).get('people') or []
                if people:
                    throws = (people[0].get('pitchHand') or {}).get('code')
            except Exception:
                pass
        if isinstance(throws, str) and throws.upper().startswith('L'):
            throws_cache[key] = 'L'
        else:
            throws_cache[key] = 'R'
        return throws_cache[key]
    except Exception:
        return 'R'


def main() -> int:
    players_path = Path(PLAYERS_FILE)
    if not players_path.exists():
        raise SystemExit(f"Players file not found: {players_path}")

    with players_path.open('r', encoding='utf-8') as f:
        players_data = json.load(f)

    # Ensure folder label present top-level
    mutated_top_level = False
    if FOLDER_LABEL:
        folders_list = players_data.setdefault('folders', [])
        if isinstance(folders_list, list) and FOLDER_LABEL not in folders_list:
            folders_list.append(FOLDER_LABEL)
            mutated_top_level = True

    existing_pitchers = players_data.get('pitchers', [])
    existing_keys: Set[str] = {normalize_existing_name(p['name']) for p in existing_pitchers if 'name' in p}

    # Query using the lower (reliever) threshold so we can apply a stricter starter filter locally
    df = pitching_stats(YEAR, qual=MIN_IP)
    if TEAM:
        if 'Team' not in df.columns:
            print(f"Warning: TEAM specified ({TEAM}) but 'Team' column not present; ignoring filter.")
        else:
            before = len(df)
            df = df[df['Team'] == TEAM].copy()
            print(f"Filtered to team {TEAM}: {len(df)} rows (from {before}).")

    candidates: List[Dict] = []
    for _, row in df.iterrows():
        name = row['Name'] if 'Name' in row else row.get('Player')
        if not name:
            continue
        base_key = normalize_existing_name(name)
        if base_key in existing_keys:
            continue
        pitcher_type = classify_pitcher(row)
        # innings pitched may be in 'IP' (can be x.y where .1 = 1/3 inning). We'll convert to float by innings = int + (decimal*10)/3
        ip_raw = row.get('IP', 0)
        innings = 0.0
        try:
            if isinstance(ip_raw, (int, float)):
                innings = float(ip_raw)
            else:
                # string like '123.2'
                s = str(ip_raw)
                if '.' in s:
                    whole, frac = s.split('.')
                    innings = float(whole) + (int(frac) / 10.0) * (1/3*10)  # approximate; better: (int(frac)/10)*(1/3)
                    # refine properly
                    innings = float(whole) + (int(frac) * (1/3))
                else:
                    innings = float(s)
        except Exception:
            innings = 0.0

        # Apply role-specific minimum IP thresholds
        if pitcher_type == 'SP':
            if innings < MIN_IP_SP:
                continue
        else:  # RP
            if innings < MIN_IP:
                continue
        stamina = compute_stamina(pitcher_type, innings)
        throws = fetch_throws(name)
        entry = build_pitcher_entry(row, pitcher_type, throws, stamina)
        if FOLDER_LABEL:
            entry['folders'] = [FOLDER_LABEL]
        candidates.append(entry)

    if LIMIT > 0:
        candidates = candidates[:LIMIT]

    if not candidates:
        if mutated_top_level:
            if DRY_RUN:
                print(f"Dry run: would have added folder '{FOLDER_LABEL}' (no new pitchers).")
            else:
                with players_path.open('w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
                print(f"Added folder '{FOLDER_LABEL}' (no new pitchers).")
        else:
            print('No new pitchers to add.')
        return 0

    for c in candidates:
        print(f" - {c['name']} Type={c['pitcher_type']} K%={c['k_pct']} BB%={c['bb_pct']} Throws={c['throws']} Stamina={c['stamina']}")

    if DRY_RUN:
        print('\nDry run enabled; no file written.')
        return 0

    players_data.setdefault('pitchers', []).extend(candidates)
    with players_path.open('w', encoding='utf-8') as f:
        json.dump(players_data, f, ensure_ascii=False, indent=2)
    print(f"Updated {players_path} with {len(candidates)} new pitchers (folder tag='{FOLDER_LABEL}').")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
