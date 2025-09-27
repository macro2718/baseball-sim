import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from baseball_sim.config import get_project_paths, path_manager, setup_project_environment
from baseball_sim.gameplay.game import GameState
from baseball_sim.gameplay.statistics import StatsCalculator
from baseball_sim.gameplay.substitutions import SubstitutionManager
from baseball_sim.gameplay.cpu_strategy import (
    CPUPlayType,
    plan_pitcher_change,
    plan_defensive_substitutions,
    plan_pinch_hit,
    plan_pinch_run,
    select_offense_play,
)


@dataclass
class TeamContext:
    """試合を跨いで必要となるチーム固有の状態"""

    team: object
    original_name: str
    unique_name: str
    pitcher_pool: List[object]
    rotation: List[object]
    rotation_index: int = 0

    def next_starting_pitcher(self) -> Optional[object]:
        pitcher = select_starting_pitcher(self.rotation, self.rotation_index)
        if self.rotation:
            self.rotation_index = (self.rotation_index + 1) % len(self.rotation)
        return pitcher

setup_project_environment()


def get_pitcher_rotation(pitchers, preferred_rotation=None):
    """登録順または指定順で先発投手のローテーションリストを作成する"""
    if not pitchers:
        return []

    if preferred_rotation:
        rotation_list = []
        seen = set()
        for pitcher in preferred_rotation:
            if pitcher and pitcher in pitchers and id(pitcher) not in seen:
                rotation_list.append(pitcher)
                seen.add(id(pitcher))
        if rotation_list:
            return rotation_list

    starters = [
        pitcher
        for pitcher in pitchers
        if getattr(pitcher, "pitcher_type", "").upper() == "SP"
    ]

    return list(starters)


def select_starting_pitcher(rotation, index):
    """ローテーションと現在のインデックスから先発投手を取得する"""
    if not rotation:
        return None

    safe_index = index % len(rotation)
    return rotation[safe_index]


def _assign_unique_team_names(contexts: Sequence[TeamContext]) -> None:
    """同名チームに対して -1, -2 ... といった識別子を付与する"""

    totals: Dict[str, int] = defaultdict(int)
    for ctx in contexts:
        totals[ctx.original_name] += 1

    counters: Dict[str, int] = defaultdict(int)
    for ctx in contexts:
        base = ctx.original_name
        if totals[base] > 1:
            counters[base] += 1
            unique = f"{base}-{counters[base]}"
        else:
            unique = base
        ctx.unique_name = unique
        if hasattr(ctx.team, "name"):
            ctx.team.name = unique


def _build_team_contexts(team_datas: Sequence[Mapping[str, object]]) -> List[TeamContext]:
    """入力データからリーグ戦用のチームコンテキストを作成"""

    if not team_datas:
        raise ValueError("League simulation requires at least two teams.")

    from baseball_sim.data.loader import DataLoader

    player_data_path = path_manager.get_players_data_path()
    player_data = DataLoader.load_json_data(player_data_path)

    contexts: List[TeamContext] = []
    for data in team_datas:
        if not isinstance(data, Mapping):
            raise ValueError("Each team configuration must be a mapping object.")
        team_config = dict(data)
        team = DataLoader.create_team(team_config, player_data=player_data)
        context = TeamContext(
            team=team,
            original_name=team_config.get("name", getattr(team, "name", "Team")),
            unique_name=getattr(team, "name", "Team"),
            pitcher_pool=list(team.pitchers),
            rotation=get_pitcher_rotation(list(team.pitchers), getattr(team, "pitcher_rotation", None)),
        )
        contexts.append(context)

    _assign_unique_team_names(contexts)
    return contexts


def _initialize_league_results(
    contexts: Sequence[TeamContext],
    *,
    role_assignment: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    """リーグシミュレーション結果の基本構造を初期化する"""

    results: Dict[str, object] = {
        "games": [],
        "team_stats": {},
        "teams": {},
        "players": {},
        "pitchers": {},
        "meta": {},
        "team_aliases": {},
    }

    for index, ctx in enumerate(contexts):
        team = ctx.team
        team_name = getattr(team, "name", f"Team {index + 1}")
        original_name = ctx.original_name

        results["teams"][team_name] = team
        if original_name not in results["teams"]:
            results["teams"][original_name] = team

        # 役割（home/awayなど）に対応するエイリアスを登録
        if role_assignment:
            for role, assigned_index in role_assignment.items():
                if assigned_index != index:
                    continue
                results["teams"][role] = team
                suffix = "Home" if role == "home" else "Away"
                alias_name = f"{original_name} ({suffix})"
                results["teams"][alias_name] = team
                results["team_aliases"][alias_name] = team_name

        # 選手・投手をチームごとに格納（同名選手の衝突を避ける）
        for collection in (
            getattr(team, "lineup", []) or [],
            getattr(team, "bench", []) or [],
        ):
            for player in collection:
                if player is None:
                    continue
                unique_key = f"{team_name}:{getattr(player, 'name', 'Player')}"
                results["players"][unique_key] = player
                results["players"].setdefault(getattr(player, "name", unique_key), player)

        for pitcher in ctx.pitcher_pool:
            if pitcher is None:
                continue
            unique_key = f"{team_name}:{getattr(pitcher, 'name', 'Pitcher')}"
            results["pitchers"][unique_key] = pitcher
            results["pitchers"].setdefault(getattr(pitcher, "name", unique_key), pitcher)

    return results


def _generate_round_robin_pairs(num_teams: int) -> List[List[Tuple[int, int]]]:
    """偶数チーム数の総当たり組み合わせを生成する"""

    if num_teams % 2 != 0:
        raise ValueError("Team count must be even for league scheduling.")

    if num_teams < 2:
        return []

    indices = list(range(num_teams))
    half = num_teams // 2
    rounds: List[List[Tuple[int, int]]] = []

    for round_index in range(num_teams - 1):
        pairs: List[Tuple[int, int]] = []
        for i in range(half):
            first = indices[i]
            second = indices[-(i + 1)]
            if round_index % 2 == 0:
                home, away = first, second
            else:
                home, away = second, first
            pairs.append((home, away))
        rounds.append(pairs)

        # circle method rotation (固定1チーム + 残りを回転)
        indices = [indices[0]] + [indices[-1]] + indices[1:-1]

    return rounds


def _simulate_league(
    *,
    team_datas: Sequence[Mapping[str, object]],
    games_per_card: int,
    cards_per_opponent: int,
    progress_callback=None,
    message_callback=None,
    role_assignment: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    """リーグ戦全体をシミュレーションして結果を返す"""

    if games_per_card <= 0:
        raise ValueError("games_per_card must be greater than zero.")
    if cards_per_opponent <= 0:
        raise ValueError("cards_per_opponent must be greater than zero.")

    contexts = _build_team_contexts(team_datas)
    if len(contexts) % 2 != 0:
        raise ValueError("League simulation requires an even number of teams.")

    results = _initialize_league_results(contexts, role_assignment=role_assignment)

    rounds = _generate_round_robin_pairs(len(contexts))
    total_days = (len(contexts) - 1) * games_per_card * cards_per_opponent
    total_games = total_days * (len(contexts) // 2)

    results["meta"].update(
        {
            "total_teams": len(contexts),
            "games_per_card": games_per_card,
            "cards_per_opponent": cards_per_opponent,
            "total_days": total_days,
            "scheduled_games": total_games,
        }
    )

    day_number = 1
    games_played = 0

    for card_index in range(cards_per_opponent):
        for round_index, round_pairs in enumerate(rounds):
            for repeat_index in range(games_per_card):
                if message_callback:
                    message_callback(
                        "Day {day}: card {card}/{cards}, round {round}/{rounds}, game {game}/{games}".format(
                            day=day_number,
                            card=card_index + 1,
                            cards=cards_per_opponent,
                            round=round_index + 1,
                            rounds=len(rounds),
                            game=repeat_index + 1,
                            games=games_per_card,
                        )
                    )

                for home_index, away_index in round_pairs:
                    home_ctx = contexts[home_index]
                    away_ctx = contexts[away_index]

                    home_starter = home_ctx.next_starting_pitcher()
                    away_starter = away_ctx.next_starting_pitcher()

                    reset_team_and_players(
                        home_ctx.team,
                        away_ctx.team,
                        home_pitcher_pool=home_ctx.pitcher_pool,
                        away_pitcher_pool=away_ctx.pitcher_pool,
                        home_starting_pitcher=home_starter,
                        away_starting_pitcher=away_starter,
                    )

                    game = GameState(home_ctx.team, away_ctx.team)
                    game_result = simulate_single_game(game)
                    game_result.update(
                        {
                            "day": day_number,
                            "card": card_index + 1,
                            "round": round_index + 1,
                            "card_game": repeat_index + 1,
                        }
                    )
                    results["games"].append(game_result)
                    update_statistics(results, game, game_result)
                    games_played += 1

                if progress_callback:
                    progress_callback(day_number, total_days)

                day_number += 1

    results["meta"]["completed_games"] = games_played
    results["meta"]["completed_days"] = day_number - 1

    return results


def _iter_unique_teams(results: Mapping[str, object]) -> Iterable[object]:
    """結果に含まれる重複しないチームオブジェクトを列挙する"""

    teams = results.get("teams") or {}
    seen_ids: set[int] = set()
    for team in teams.values():
        if team is None:
            continue
        identifier = id(team)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        yield team


def simulate_games(
    num_games=10,
    output_file=None,
    progress_callback=None,
    message_callback=None,
    *,
    home_team_data=None,
    away_team_data=None,
    save_to_file=True,
    league_options: Optional[Mapping[str, object]] = None,
):
    """試合を複数日程でシミュレートし、結果を返す"""

    # 出力ファイルの設定
    output_path = None
    if save_to_file:
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_root = get_project_paths().project_root
            output_dir = os.path.join(project_root, "simulation_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"simulation_results_{timestamp}.txt")
        else:
            output_path = output_file

        message = f"Results will be saved to {output_path}."
        if message_callback:
            message_callback(message)
        else:
            print(message)

    # リーグ構成を決定
    if league_options is not None:
        team_datas = list(league_options.get("teams", []))
        games_per_card = int(league_options.get("games_per_card", 0) or 0)
        cards_per_opponent = int(league_options.get("cards_per_opponent", 0) or 0)
        role_assignment = league_options.get("role_assignment")
    else:
        # 従来仕様: 指定の2チームでシリーズを構成
        from baseball_sim.data.loader import DataLoader

        team_data_path = path_manager.get_teams_data_path()
        raw_team_data = DataLoader.load_json_data(team_data_path)
        if not isinstance(raw_team_data, Mapping):
            raise ValueError("Team data file is invalid or missing required structure.")

        base_home = home_team_data or raw_team_data.get("home_team")
        base_away = away_team_data or raw_team_data.get("away_team")
        if not isinstance(base_home, Mapping) or not isinstance(base_away, Mapping):
            raise ValueError("Home/Away team data could not be loaded.")

        team_datas = [dict(base_home), dict(base_away)]
        games_per_card = int(num_games) if num_games else 0
        cards_per_opponent = 1
        role_assignment = {"home": 0, "away": 1}

    if games_per_card <= 0 or cards_per_opponent <= 0:
        raise ValueError("リーグシミュレーションには1試合以上の設定が必要です。")

    results = _simulate_league(
        team_datas=team_datas,
        games_per_card=games_per_card,
        cards_per_opponent=cards_per_opponent,
        progress_callback=progress_callback,
        message_callback=message_callback,
        role_assignment=role_assignment,
    )

    # 結果をファイルに出力
    if save_to_file and output_path:
        output_results(results, output_path)

    results["output_file"] = output_path

    if save_to_file and output_path:
        completion_message = f"Simulation complete. Results will be saved to {output_path}."
    else:
        completion_message = "Simulation complete."

    if message_callback:
        message_callback(completion_message)
    else:
        print(completion_message)

    return results

def reset_team_and_players(
    home_team,
    away_team,
    *,
    home_pitcher_pool=None,
    away_pitcher_pool=None,
    home_starting_pitcher=None,
    away_starting_pitcher=None,
):
    """チームと選手の状態をリセットする（投手のスタミナなど）"""
    # 1試合ごとに退場(ejected)扱いをリセットし、シリーズ中に再登板/再出場できるようにする
    # これを行わないと、交代した投手が以後の試合で永続的に使用不可となり、登板数が極端に偏る。
    for team in (home_team, away_team):
        if hasattr(team, "ejected_players"):
            team.ejected_players = []

    if home_pitcher_pool is not None:
        home_team.pitchers = list(home_pitcher_pool)
    if away_pitcher_pool is not None:
        away_team.pitchers = list(away_pitcher_pool)

    if hasattr(home_team, "pitcher_rotation"):
        home_team.pitcher_rotation = [
            pitcher for pitcher in home_team.pitcher_rotation if pitcher in home_team.pitchers
        ]
    if hasattr(away_team, "pitcher_rotation"):
        away_team.pitcher_rotation = [
            pitcher for pitcher in away_team.pitcher_rotation if pitcher in away_team.pitchers
        ]

    # 各チームの投手のスタミナを調整
    # - SP: 毎試合フルにリセット（従来通り）
    # - RP: 試合間でスタミナを引き継ぎ、試合前に+10回復（上限は基礎スタミナ）
    for pitcher in home_team.pitchers:
        ptype = getattr(pitcher, "pitcher_type", "").upper()
        if ptype == "RP":
            current = getattr(pitcher, "current_stamina", pitcher.stamina)
            pitcher.current_stamina = min(pitcher.stamina, current + 10)
        else:
            pitcher.current_stamina = pitcher.stamina

    for pitcher in away_team.pitchers:
        ptype = getattr(pitcher, "pitcher_type", "").upper()
        if ptype == "RP":
            current = getattr(pitcher, "current_stamina", pitcher.stamina)
            pitcher.current_stamina = min(pitcher.stamina, current + 10)
        else:
            pitcher.current_stamina = pitcher.stamina

    # 打順をリセット
    home_team.current_batter_index = 0
    away_team.current_batter_index = 0
    
    # 先発投手をリセット（必要に応じて）
    home_team.current_pitcher = (
        home_starting_pitcher
        if home_starting_pitcher is not None
        else (home_team.pitchers[0] if home_team.pitchers else None)
    )
    away_team.current_pitcher = (
        away_starting_pitcher
        if away_starting_pitcher is not None
        else (away_team.pitchers[0] if away_team.pitchers else None)
    )

def simulate_single_game(game):
    """1試合をCPU対CPUのロジックでシミュレーション実行し、結果を返す"""
    game_result = {
        "home_team": game.home_team.name,
        "away_team": game.away_team.name,
        "home_score": 0,
        "away_score": 0,
        "innings": 0,
        "events": [],
    }

    # コンテキストを使って同一盤面での重複CPU判断を抑制
    last_context = None

    appeared_pitchers = set()

    while not game.game_ended:
        batting_team = game.batting_team
        fielding_team = game.fielding_team
        batter = batting_team.current_batter
        pitcher = fielding_team.current_pitcher

        if batter is None or pitcher is None:
            # 何らかの不整合（交代直後等）に備えて安全に抜ける
            break

        # 登板数(G)のカウント: この試合で初めて登板した投手のみ+1
        pid = id(pitcher)
        if pid not in appeared_pitchers:
            appeared_pitchers.add(pid)
            try:
                pitcher.pitching_stats["G"] = pitcher.pitching_stats.get("G", 0) + 1
            except Exception:
                pass

        # 盤面コンテキストが変わったときのみ、CPUの守備采配/代打・代走検討を行う
        base_sig = tuple(1 if r is not None else 0 for r in game.bases[:3])
        context = (game.last_play.get("sequence") if game.last_play else None, game.inning, game.is_top_inning, game.outs, base_sig)
        if context != last_context:
            last_context = context

            # 守備側CPU: 投手交代・守備交代
            def apply_defense_plans():
                sub = SubstitutionManager(fielding_team)
                # 先に守備交代（適性/守備固め）を適用
                try:
                    dplans = list(plan_defensive_substitutions(game, fielding_team, sub))
                except Exception:
                    dplans = []
                if dplans:
                    bench = sub.get_available_bench_players()
                    for plan in dplans:
                        # 現在のベンチから対象選手のインデックスを検索
                        try:
                            bench_index = bench.index(plan.bench_player)
                        except ValueError:
                            continue
                        sub.execute_defensive_substitution(bench_index, plan.lineup_index)
                # 投手交代
                try:
                    pplan = plan_pitcher_change(game, fielding_team, sub)
                except Exception:
                    pplan = None
                if pplan is not None:
                    sub.execute_pitcher_change(pplan.pitcher_index)

            apply_defense_plans()

            # 攻撃側CPU: 代打・代走の検討
            try:
                offense_sub = SubstitutionManager(batting_team)
                ph_plan = plan_pinch_hit(game, batting_team, offense_sub)
            except Exception:
                ph_plan = None
            if ph_plan is not None:
                offense_sub.execute_pinch_hit(ph_plan.bench_index, ph_plan.lineup_index)

            try:
                pr_plan = plan_pinch_run(game, batting_team, offense_sub)
            except Exception:
                pr_plan = None
            if pr_plan is not None:
                success, _ = offense_sub.execute_defensive_substitution(
                    pr_plan.bench_index, pr_plan.lineup_index
                )
                if success:
                    # ベース上の走者を差し替える
                    try:
                        new_runner = batting_team.lineup[pr_plan.lineup_index]
                        game.bases[pr_plan.base_index] = new_runner
                    except Exception:
                        pass

        # 攻撃側のプレー選択
        decision = select_offense_play(game, batting_team)

        # スコア差分の計算用に事前スコアを保持
        pre_home, pre_away = game.home_score, game.away_score
        prev_inning, prev_half = game.inning, game.is_top_inning

        action_label = None
        result_key = None
        message = ""

        if decision.play is CPUPlayType.SQUEEZE:
            # 条件を満たさなければ通常打撃
            if game.can_squeeze():
                action_label = "squeeze"
                message = game.execute_squeeze(batter, pitcher)
            else:
                decision = select_offense_play(game, batting_team)

        if result_key is None and decision.play is CPUPlayType.BUNT:
            if game.can_bunt():
                action_label = "bunt"
                message = game.execute_bunt(batter, pitcher)
            else:
                decision = select_offense_play(game, batting_team)

        if result_key is None and decision.play is CPUPlayType.STEAL:
            # 盗塁は打席結果を伴わない進行
            if game.can_steal():
                action_label = "steal"
                info = game.execute_steal()
                result_key = info.get("result")
                message = info.get("message", "")
            else:
                decision = select_offense_play(game, batting_team)

        if result_key is None:
            # 通常打撃
            action_label = action_label or "swing"
            result_key = game.calculate_result(batter, pitcher)
            message = game.apply_result(result_key, batter)

        # イベントの集計
        inning_changed = (prev_inning != game.inning) or (prev_half != game.is_top_inning)
        if action_label in ("swing", "bunt", "squeeze"):
            # 打席を消費するアクションでは投手スタミナ減少
            pitcher.decrease_stamina()
            if not inning_changed:
                game.batting_team.next_batter()

        # 盗塁では打者は据え置き、イニング変化時はそのまま

        # ランの差分（打者側に入った得点）を算出
        post_home, post_away = game.home_score, game.away_score
        if batting_team is game.home_team:
            runs_scored = max(0, post_home - pre_home)
        else:
            runs_scored = max(0, post_away - pre_away)

        event = {
            "inning": f"{prev_inning} {'Top' if prev_half else 'Bottom'}",
            "batter": getattr(batter, 'name', 'Unknown'),
            "pitcher": getattr(pitcher, 'name', 'Unknown'),
            "action": action_label,
            "result": result_key,
            "message": message,
            "runs_scored": runs_scored,
            "batting_team": getattr(batting_team, 'name', ''),
            "fielding_team": getattr(fielding_team, 'name', ''),
        }
        game_result["events"].append(event)

        # 終了判定は GameState に委譲（サヨナラ/延長制限を含む）

    # 最終結果を更新
    game_result["home_score"] = game.home_score
    game_result["away_score"] = game.away_score
    game_result["innings"] = game.inning

    return game_result

def update_statistics(results, game, game_result):
    """チームの勝敗統計を更新する"""

    def ensure_entry(team_obj):
        team_name = getattr(team_obj, "name", "Team")
        stats = results.setdefault("team_stats", {}).setdefault(
            team_name,
            {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "runs_scored": 0,
                "runs_allowed": 0,
                "games": 0,
            },
        )
        return team_name, stats

    home_name, home_stats = ensure_entry(game.home_team)
    away_name, away_stats = ensure_entry(game.away_team)

    home_stats["games"] += 1
    away_stats["games"] += 1

    if game.home_score > game.away_score:
        home_stats["wins"] += 1
        away_stats["losses"] += 1
    elif game.away_score > game.home_score:
        home_stats["losses"] += 1
        away_stats["wins"] += 1
    else:
        home_stats["draws"] += 1
        away_stats["draws"] += 1

    home_stats["runs_scored"] += game.home_score
    home_stats["runs_allowed"] += game.away_score
    away_stats["runs_scored"] += game.away_score
    away_stats["runs_allowed"] += game.home_score

    # home/awayなどのエイリアスにも同じ参照を割り当てる
    aliases = results.get("team_aliases", {})
    for alias, target in aliases.items():
        if target == home_name:
            results["team_stats"][alias] = home_stats
        elif target == away_name:
            results["team_stats"][alias] = away_stats

def output_results(results, output_file):
    """シミュレーション結果をファイルに出力する"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("===== シミュレーション結果 =====\n\n")

        meta = results.get("meta", {})
        if meta:
            f.write("===== リーグ情報 =====\n")
            if "total_teams" in meta:
                f.write(f"チーム数: {meta['total_teams']}\n")
            if "games_per_card" in meta and "cards_per_opponent" in meta:
                f.write(
                    f"1カードあたりの試合数: {meta['games_per_card']} / 対戦カード回数: {meta['cards_per_opponent']}\n"
                )
            if "completed_days" in meta or "total_days" in meta:
                f.write(
                    f"総日数: {meta.get('completed_days', meta.get('total_days', 0))}"
                    f" / 予定日数: {meta.get('total_days', meta.get('completed_days', 0))}\n"
                )
            if "completed_games" in meta or "scheduled_games" in meta:
                f.write(
                    f"総試合数: {meta.get('completed_games', meta.get('scheduled_games', 0))}"
                    f" / 予定試合数: {meta.get('scheduled_games', meta.get('completed_games', 0))}\n"
                )
            f.write("\n")

        unique_teams = list(_iter_unique_teams(results))
        
        # チーム成績の基本出力
        f.write("===== チーム成績 =====\n")
        f.write(f"{'チーム名':<15} {'試合':<5} {'勝':<5} {'負':<5} {'分':<5} {'得点':<5} {'失点':<5} {'勝率':<5}\n")
        f.write("-" * 60 + "\n")

        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            stats = results.get("team_stats", {}).get(team_name)
            if not stats:
                continue
            games_played = int(
                stats.get("games")
                or (stats.get("wins", 0) + stats.get("losses", 0) + stats.get("draws", 0))
            )
            win_pct = stats.get("wins", 0) / games_played if games_played > 0 else 0

            f.write(
                f"{team_name:<15} {games_played:<5} {stats.get('wins', 0):<5} {stats.get('losses', 0):<5} {stats.get('draws', 0):<5} "
                f"{stats.get('runs_scored', 0):<5} {stats.get('runs_allowed', 0):<5} {win_pct:.3f}\n"
            )

        # チーム打撃成績の出力
        f.write("\n===== チーム打撃成績 =====\n")
        f.write(f"{'チーム名':<15} {'打率':<5} {'OBP':<5} {'SLG':<5} {'OPS':<5} {'1B計':<5} {'2B計':<5} {'3B計':<5} {'HR計':<5} "
               f"{'三振':<5} {'四球':<5}\n")
        f.write("-" * 80 + "\n")
        
        # チーム打撃成績を計算して出力
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            # チーム打撃成績の集計
            team_pa = 0
            team_ab = 0
            team_hits = 0
            team_bb = 0
            team_hr = 0
            team_so = 0
            team_singles = 0
            team_doubles = 0
            team_triples = 0
            
            # チーム全体の打撃成績を集計
            for player in team.lineup:
                team_pa += player.stats["PA"]
                team_ab += player.stats["AB"]
                team_singles += player.stats["1B"]
                team_doubles += player.stats.get("2B", 0)
                team_triples += player.stats.get("3B", 0)
                team_hr += player.stats["HR"]
                team_hits += player.stats["1B"] + player.stats["2B"] + player.stats["3B"] + player.stats["HR"]
                team_bb += player.stats["BB"]
                team_so += player.stats["SO"]
                
            # チーム打率計算
            team_avg = team_hits / team_ab if team_ab > 0 else 0
            
            # チーム出塁率計算（PA未集計のため、OBP = (H + BB) / (AB + BB) と等価の簡略式を使用）
            team_obp = StatsCalculator.calculate_obp(team_hits, team_bb, team_ab)
            
            # チーム長打率計算（修正版）
            singles = team_hits - team_doubles - team_triples - team_hr
            total_bases = singles + (team_doubles * 2) + (team_triples * 3) + (team_hr * 4)
            team_slg = total_bases / team_ab if team_ab > 0 else 0
            
            # チームOPS計算
            team_ops = team_obp + team_slg
            
            # チーム打撃成績の出力
            f.write(f"{team_name:<15} {team_avg:.3f} {team_obp:.3f} {team_slg:.3f} {team_ops:.3f} "
                   f"{team_singles:<5} {team_doubles:<5} {team_triples:<5} {team_hr:<5} "
                   f"{team_so:<5} {team_bb:<5}\n")
        
        # チーム投手成績の出力
        f.write("\n===== チーム投手成績 =====\n")
        f.write(f"{'チーム名':<15} {'防御率':<6} {'WHIP':<6} {'K/9':<6} {'BB/9':<6}\n")
        f.write("-" * 60 + "\n")
        
        # チーム投手成績を計算して出力
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            # チーム投手成績の集計
            team_ip = 0
            team_h = 0
            team_bb_allowed = 0
            team_so_pitched = 0
            team_er = 0
            
            # チーム全体の投手成績を集計
            for pitcher in team.pitchers:
                team_ip += pitcher.pitching_stats["IP"]
                team_h += pitcher.pitching_stats["H"]
                team_bb_allowed += pitcher.pitching_stats["BB"]
                team_so_pitched += pitcher.pitching_stats["SO"]
                team_er += pitcher.pitching_stats["ER"]
            
            # チーム防御率計算
            team_era = (team_er * 9) / team_ip if team_ip > 0 else 0
            
            # チームWHIP計算
            team_whip = (team_h + team_bb_allowed) / team_ip if team_ip > 0 else 0
            
            # チームK/9計算
            team_k_per_9 = (team_so_pitched * 9) / team_ip if team_ip > 0 else 0
            
            # チームBB/9計算
            team_bb_per_9 = (team_bb_allowed * 9) / team_ip if team_ip > 0 else 0
            
            # チーム投手成績の出力
            f.write(f"{team_name:<15} {team_era:<6.2f} {team_whip:<6.2f} {team_k_per_9:<6.2f} {team_bb_per_9:<6.2f}\n")

        # チームごとの選手成績の出力
        f.write("\n===== チーム詳細成績 =====\n")
        f.write(f"{'チーム名':<15} {'打率':<4} {'OBP':<5} {'SLG':<4} {'OPS':<4} {'1B計':<4} {'2B計':<4} {'3B計':<5} {'HR計':<4} "
               f"{'三振':<3} {'四球':<3} {'防御率':<3} {'WHIP':<3} {'K/9':<3} {'BB/9':<5}\n")
        f.write("-" * 120 + "\n")
        
        # チーム詳細成績を計算して出力
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            # チーム打撃成績の集計
            team_pa = 0
            team_ab = 0
            team_hits = 0
            team_bb = 0
            team_hr = 0
            team_so = 0
            team_singles = 0
            team_doubles = 0
            team_triples = 0
            
            # チーム全体の打撃成績を集計
            for player in team.lineup:
                team_pa += player.stats["PA"]
                team_ab += player.stats["AB"]
                team_singles += player.stats["1B"]
                team_doubles += player.stats.get("2B", 0)
                team_triples += player.stats.get("3B", 0)
                team_hr += player.stats["HR"]
                team_hits += player.stats["1B"] + player.stats["2B"] + player.stats["3B"] + player.stats["HR"]
                team_bb += player.stats["BB"]
                team_so += player.stats["SO"]
                
            # チーム打率計算
            team_avg = team_hits / team_ab if team_ab > 0 else 0
            
            # チーム出塁率計算（簡略式）
            team_obp = StatsCalculator.calculate_obp(team_hits, team_bb, team_ab)
            
            # チーム長打率計算（修正版）
            singles = team_hits - team_doubles - team_triples - team_hr
            total_bases = singles + (team_doubles * 2) + (team_triples * 3) + (team_hr * 4)
            team_slg = total_bases / team_ab if team_ab > 0 else 0
            
            # チームOPS計算
            team_ops = team_obp + team_slg
            
            # チーム投手成績の集計
            team_ip = 0
            team_h = 0
            team_bb_allowed = 0
            team_so_pitched = 0
            team_er = 0
            
            # チーム全体の投手成績を集計
            for pitcher in team.pitchers:
                team_ip += pitcher.pitching_stats["IP"]
                team_h += pitcher.pitching_stats["H"]
                team_bb_allowed += pitcher.pitching_stats["BB"]
                team_so_pitched += pitcher.pitching_stats["SO"]
                team_er += pitcher.pitching_stats["ER"]
            
            # チーム防御率計算
            team_era = (team_er * 9) / team_ip if team_ip > 0 else 0
            
            # チームWHIP計算
            team_whip = (team_h + team_bb_allowed) / team_ip if team_ip > 0 else 0
            
            # チームK/9計算
            team_k_per_9 = (team_so_pitched * 9) / team_ip if team_ip > 0 else 0
            
            # チームBB/9計算
            team_bb_per_9 = (team_bb_allowed * 9) / team_ip if team_ip > 0 else 0
            
            # チーム詳細成績の出力（ヒット種類ごとの合計を追加）
            f.write(f"{team_name:<17} {team_avg:.3f} {team_obp:.3f} {team_slg:.3f} {team_ops:.3f} "
                   f"{team_singles:<5} {team_doubles:<5} {team_triples:<5} {team_hr:<5} "
                   f"{team_so:<5} {team_bb:<5} "
                   f"{team_era:.2f} {team_whip:.2f} {team_k_per_9:.2f} {team_bb_per_9:.2f}\n")
        
        # チームごとの選手成績の出力
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            f.write(f"\n===== {team_name} 選手成績 =====\n")
            
            # 打者成績
            f.write("\n【打者成績】\n")
            f.write(f"{'選手名':<18} {'PA':<4} {'AB':<4} {'1B':<4} {'2B':<4} {'3B':<4} {'HR':<4} {'RBI':<4} {'BB':<4} {'K':<4} "
                  f"{'AVG':<6} {'OBP':<5} {'SLG':<5} {'OPS':<5} {'K%':<5} {'BB%':<5}\n")
            f.write("-" * 130 + "\n")
            
            # チーム選手の打撃成績を表示
            for player in team.lineup:
                # 選手の成績を取得
                pa = player.stats["PA"]
                ab = player.stats["AB"]
                singles = player.stats["1B"]
                doubles = player.stats.get("2B", 0)
                triples = player.stats.get("3B", 0)
                hr = player.stats["HR"]
                rbi = player.stats["RBI"]
                bb = player.stats["BB"]
                so = player.stats["SO"]
                
                # 指標を計算
                avg = player.get_avg()
                obp = player.get_obp()
                slg = player.get_slg()
                ops = player.get_ops()
                
                # K%とBB%を計算
                k_pct = (so / pa * 100) if pa > 0 else 0
                bb_pct = (bb / pa * 100) if pa > 0 else 0
                
                # 打者成績を出力
                f.write(f"{player.name:<20} {pa:<4} {ab:<4} {singles:<4} {doubles:<4} {triples:<4} "
                      f"{hr:<4} {rbi:<4} {bb:<4} {so:<4} "
                      f"{avg:.3f} {obp:.3f} {slg:.3f} {ops:.3f} {k_pct:.1f} {bb_pct:.1f}\n")
            
            # 投手成績
            f.write("\n【投手成績】\n")
            f.write(f"{'選手名':<15} {'IP':<5} {'H':<4} {'BB':<4} {'K':<4} {'R':<4} {'ER':<4} {'HR':<4} "
                  f"{'ERA':<5} {'WHIP':<5} {'K/9':<5} {'BB/9':<5} {'HR/9':<5}\n")
            f.write("-" * 120 + "\n")
            
            # チーム選手の投手成績を表示
            for pitcher in team.pitchers:
                # 投手の成績を取得
                ip = pitcher.pitching_stats["IP"]
                h = pitcher.pitching_stats["H"]
                bb = pitcher.pitching_stats["BB"]
                so = pitcher.pitching_stats["SO"]
                r = pitcher.pitching_stats["R"]
                er = pitcher.pitching_stats["ER"]
                hr = pitcher.pitching_stats["HR"]
                
                # 指標を計算
                era = pitcher.get_era()
                whip = pitcher.get_whip()
                k_per_9 = pitcher.get_k_per_9()
                bb_per_9 = pitcher.get_bb_per_9()
                
                # HR/9を計算
                hr_per_9 = (hr * 9) / ip if ip > 0 else 0
                
                # 投手成績を出力
                f.write(f"{pitcher.name:<15} {ip:<5.1f} {h:<4} {bb:<4} {so:<4} {r:<4} {er:<4} {hr:<4} "
                      f"{era:<5.2f} {whip:<5.2f} {k_per_9:<5.2f} {bb_per_9:<5.2f} {hr_per_9:<5.2f}\n")
        
        # 全選手の成績
        f.write("\n===== 全選手成績 =====\n")
        
        # 打者成績の出力
        f.write("\n===== 打者成績 =====\n")
        f.write(f"{'選手名':<18} {'チーム':<10} {'PA':<4} {'AB':<4} {'1B':<4} {'2B':<4} {'3B':<4} {'HR':<4} {'RBI':<4} {'BB':<4} {'K':<4} "
              f"{'AVG':<6} {'OBP':<5} {'SLG':<5} {'OPS':<5} {'K%':<5} {'BB%':<5}\n")
        f.write("-" * 140 + "\n")
        
        # チームに所属する選手をキーとして保存
        player_team: Dict[int, str] = {}
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            for player in team.lineup:
                player_team[id(player)] = team_name
        
        # すべての打者の成績を出力
        seen_player_ids: set[int] = set()
        for name, player in results["players"].items():
            # 選手の成績を取得
            pid = id(player)
            if pid in seen_player_ids:
                continue
            seen_player_ids.add(pid)

            pa = player.stats["PA"]
            
            # 打席がゼロなら表示しない
            if pa == 0:
                continue
                
            ab = player.stats["AB"]
            singles = player.stats["1B"]
            doubles = player.stats.get("2B", 0)
            triples = player.stats.get("3B", 0)
            hr = player.stats["HR"]
            rbi = player.stats["RBI"]
            bb = player.stats["BB"]
            so = player.stats["SO"]
            
            # 指標を計算
            avg = player.get_avg()
            obp = player.get_obp()
            slg = player.get_slg()
            ops = player.get_ops()
            
            # K%とBB%を計算
            k_pct = (so / pa * 100) if pa > 0 else 0
            bb_pct = (bb / pa * 100) if pa > 0 else 0
            
            team_name = player_team.get(pid, "不明")
            
            # 打者成績を出力
            f.write(f"{name:<20} {team_name:<10} {pa:<4} {ab:<4} {singles:<4} {doubles:<4} {triples:<4} "
                  f"{hr:<4} {rbi:<4} {bb:<4} {so:<4} "
                  f"{avg:.3f} {obp:.3f} {slg:.3f} {ops:.3f} {k_pct:.1f} {bb_pct:.1f}\n")
        
        # 投手成績の出力
        f.write("\n===== 投手成績 =====\n")
        f.write(f"{'選手名':<15} {'チーム':<10} {'IP':<5} {'H':<4} {'BB':<4} {'K':<4} {'R':<4} {'ER':<4} {'HR':<4} "
              f"{'ERA':<5} {'WHIP':<5} {'K/9':<5} {'BB/9':<5} {'HR/9':<5}\n")
        f.write("-" * 120 + "\n")
        
        # チームに所属する投手をキーとして保存
        pitcher_team: Dict[int, str] = {}
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            for pitcher in team.pitchers:
                pitcher_team[id(pitcher)] = team_name
        
        # すべての投手の成績を出力
        seen_pitcher_ids: set[int] = set()
        for name, pitcher in results["pitchers"].items():
            # 投手の成績を取得
            pid = id(pitcher)
            if pid in seen_pitcher_ids:
                continue
            seen_pitcher_ids.add(pid)

            ip = pitcher.pitching_stats["IP"]
            
            # 投球イニングがゼロなら表示しない
            if ip == 0:
                continue
                
            h = pitcher.pitching_stats["H"]
            bb = pitcher.pitching_stats["BB"]
            so = pitcher.pitching_stats["SO"]
            r = pitcher.pitching_stats["R"]
            er = pitcher.pitching_stats["ER"]
            hr = pitcher.pitching_stats["HR"]
            
            # 指標を計算
            era = pitcher.get_era()
            whip = pitcher.get_whip()
            k_per_9 = pitcher.get_k_per_9()
            bb_per_9 = pitcher.get_bb_per_9()
            hr_per_9 = pitcher.get_hr_per_9()
            
            team_name = pitcher_team.get(pid, "不明")
            
            # 投手成績を出力
            f.write(f"{name:<15} {team_name:<10} {ip:<5.1f} {h:<4} {bb:<4} {so:<4} {r:<4} {er:<4} {hr:<4} "
                  f"{era:<5.2f} {whip:<5.2f} {k_per_9:<5.2f} {bb_per_9:<5.2f} {hr_per_9:<5.2f}\n")
        
        # 投手登板数の出力（追加）
        f.write("\n===== 投手登板数 (G) =====\n")
        f.write(f"{'選手名':<15} {'チーム':<10} {'G':<3}\n")
        f.write("-" * 36 + "\n")
        # チームに所属する投手をキーとして保存（再利用）
        pitcher_team = {}
        for team in unique_teams:
            team_name = getattr(team, "name", "Team")
            for pitcher in team.pitchers:
                pitcher_team[id(pitcher)] = team_name
        for name, pitcher in results["pitchers"].items():
            pid = id(pitcher)
            if pid in seen_pitcher_ids:
                continue
            seen_pitcher_ids.add(pid)
            g = int(getattr(pitcher, 'pitching_stats', {}).get('G', 0) or 0)
            if g <= 0:
                continue
            team_name = pitcher_team.get(pid, "不明")
            f.write(f"{name:<15} {team_name:<10} {g:<3}\n")

        # 各試合の結果サマリー
        f.write("\n===== 試合結果サマリー =====\n")
        for i, game in enumerate(results["games"], 1):
            f.write(f"試合 {i}: {game['away_team']} {game['away_score']} - {game['home_score']} {game['home_team']} "
                  f"({game['innings']}回)\n")
