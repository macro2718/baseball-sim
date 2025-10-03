from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from baseball_sim.config import get_project_paths, setup_project_environment
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
from baseball_sim.infrastructure.logging_utils import logger as root_logger


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
PATHS = get_project_paths()

LOGGER = root_logger.getChild("interface.simulation")


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

    player_data_path = PATHS.get_players_data_path()
    player_data = DataLoader.load_json_data(player_data_path)

    contexts: List[TeamContext] = []
    for data in team_datas:
        if not isinstance(data, Mapping):
            raise ValueError("Each team configuration must be a mapping object.")
        team_config = dict(data)
        team, warnings = DataLoader.create_team(team_config, player_data=player_data)
        for warning in warnings:
            LOGGER.warning("Team setup warning: %s", warning)
        # 各試合前に初期ラインナップ/ベンチへ戻せるようスナップショットを保持
        try:
            # 初期打順と各選手の守備位置
            initial_lineup = list(getattr(team, "lineup", []) or [])
            initial_positions = [getattr(p, "current_position", None) for p in initial_lineup]
            setattr(team, "_initial_lineup", initial_lineup)
            setattr(team, "_initial_lineup_positions", initial_positions)
            # 初期ベンチ
            setattr(team, "_initial_bench", list(getattr(team, "bench", []) or []))
        except Exception:
            # 失敗時もシミュレーション継続（復元はスキップ）
            pass
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

        # 以前はファイル出力用に追加データを保持していたが、現在は不要のため収集しない

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
    """試合を複数日程でシミュレートし、結果を返す

    `output_file` および `save_to_file` は互換性維持のために残されているが、
    現在はファイルへの書き出しは行わない。
    """

    # リーグ構成を決定
    if league_options is not None:
        team_datas = list(league_options.get("teams", []))
        games_per_card = int(league_options.get("games_per_card", 0) or 0)
        cards_per_opponent = int(league_options.get("cards_per_opponent", 0) or 0)
        role_assignment = league_options.get("role_assignment")
    else:
        # 従来仕様: 指定の2チームでシリーズを構成
        from baseball_sim.data.loader import DataLoader

        team_data_path = PATHS.get_teams_data_path()
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
    # 1試合ごとに退場(ejected)扱いとラインナップ/ベンチを初期状態に戻す
    # これにより、前試合で交代・退いた野手が次試合で再出場不可になる問題を防ぐ
    for team in (home_team, away_team):
        # 退場リストのクリア
        if hasattr(team, "ejected_players"):
            team.ejected_players = []
        # ラインナップ/ベンチを初期スナップショットに復元（存在する場合）
        try:
            init_lineup = list(getattr(team, "_initial_lineup", []) or [])
            init_positions = list(getattr(team, "_initial_lineup_positions", []) or [])
            init_bench = list(getattr(team, "_initial_bench", []) or [])

            if init_lineup and init_positions and len(init_lineup) == len(init_positions):
                # ラインナップ復元
                team.lineup = list(init_lineup)
                # 守備位置マップをリセット
                if hasattr(team, "defensive_positions"):
                    for key in list(team.defensive_positions.keys()):
                        team.defensive_positions[key] = None
                # 各選手のポジションを復元
                for player, pos in zip(init_lineup, init_positions):
                    if pos is not None:
                        player.current_position = pos
                        if hasattr(team, "defensive_positions"):
                            team.defensive_positions[pos] = player
                # ベンチ復元
                team.bench = list(init_bench)
        except Exception:
            # 復元に失敗しても致命的ではないため続行
            pass

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

class CPUDecisionEngine:
    """CPU decisions around defensive/offensive adjustments for a single game."""

    def __init__(self, game):
        self.game = game

    def apply_defense_plans(self):
        """Apply defensive substitutions and pitcher changes.

        Returns:
            Dict[str, object]: Metadata describing roster changes executed.
        """

        fielding_team = self.game.fielding_team
        sub = SubstitutionManager(fielding_team)
        changes = {
            "defensive_substitutions": [],
            "pitcher_change": None,
        }

        try:
            defensive_plans = list(
                plan_defensive_substitutions(self.game, fielding_team, sub)
            )
        except Exception:
            defensive_plans = []

        if defensive_plans:
            bench = sub.get_available_bench_players()
            for plan in defensive_plans:
                try:
                    bench_index = bench.index(plan.bench_player)
                except ValueError:
                    continue
                success, message = sub.execute_defensive_substitution(
                    bench_index, plan.lineup_index
                )
                if success:
                    changes["defensive_substitutions"].append(
                        {
                            "bench_player": getattr(plan.bench_player, "name", None),
                            "lineup_index": plan.lineup_index,
                            "message": message,
                        }
                    )

        try:
            pitcher_plan = plan_pitcher_change(self.game, fielding_team, sub)
        except Exception:
            pitcher_plan = None

        if pitcher_plan is not None:
            success, message = sub.execute_pitcher_change(pitcher_plan.pitcher_index)
            if success:
                changes["pitcher_change"] = {
                    "pitcher": getattr(fielding_team.current_pitcher, "name", None),
                    "message": message,
                }

        return changes

    def apply_offense_adjustments(self):
        """Apply pinch hit/run decisions for the batting team.

        Returns:
            Dict[str, object]: Metadata describing offensive roster changes.
        """

        batting_team = self.game.batting_team
        sub = SubstitutionManager(batting_team)
        changes = {
            "pinch_hit": None,
            "pinch_run": None,
        }

        try:
            pinch_hit_plan = plan_pinch_hit(self.game, batting_team, sub)
        except Exception:
            pinch_hit_plan = None

        if pinch_hit_plan is not None:
            success, message = sub.execute_pinch_hit(
                pinch_hit_plan.bench_index, pinch_hit_plan.lineup_index
            )
            if success:
                changes["pinch_hit"] = {
                    "bench_player": getattr(
                        batting_team.lineup[pinch_hit_plan.lineup_index], "name", None
                    ),
                    "lineup_index": pinch_hit_plan.lineup_index,
                    "message": message,
                }

        try:
            pinch_run_plan = plan_pinch_run(self.game, batting_team, sub)
        except Exception:
            pinch_run_plan = None

        if pinch_run_plan is not None:
            success, message = sub.execute_defensive_substitution(
                pinch_run_plan.bench_index, pinch_run_plan.lineup_index
            )
            if success:
                try:
                    new_runner = batting_team.lineup[pinch_run_plan.lineup_index]
                    self.game.bases[pinch_run_plan.base_index] = new_runner
                except Exception:
                    pass
                changes["pinch_run"] = {
                    "runner": getattr(
                        batting_team.lineup[pinch_run_plan.lineup_index], "name", None
                    ),
                    "lineup_index": pinch_run_plan.lineup_index,
                    "base_index": pinch_run_plan.base_index,
                    "message": message,
                }

        return changes

    @staticmethod
    def assemble_event(
        *,
        game,
        batting_team,
        fielding_team,
        batter,
        pitcher,
        prev_inning,
        prev_half,
        action_label,
        result_key,
        message,
        runs_scored,
        roster_changes,
    ):
        """Create an event dictionary summarizing the play."""

        return {
            "inning": f"{prev_inning} {'Top' if prev_half else 'Bottom'}",
            "batter": getattr(batter, "name", "Unknown"),
            "pitcher": getattr(pitcher, "name", "Unknown"),
            "action": action_label,
            "result": result_key,
            "message": message,
            "runs_scored": runs_scored,
            "batting_team": getattr(batting_team, "name", ""),
            "fielding_team": getattr(fielding_team, "name", ""),
            "roster_changes": roster_changes or {},
        }


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

    decision_engine = CPUDecisionEngine(game)

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

        roster_changes = None

        # 盤面コンテキストが変わったときのみ、CPUの守備采配/代打・代走検討を行う
        base_sig = tuple(1 if r is not None else 0 for r in game.bases[:3])
        context = (
            game.last_play.get("sequence") if game.last_play else None,
            game.inning,
            game.is_top_inning,
            game.outs,
            base_sig,
        )
        if context != last_context:
            last_context = context
            defense_changes = decision_engine.apply_defense_plans()
            offense_changes = decision_engine.apply_offense_adjustments()
            roster_changes = {
                "defense": defense_changes,
                "offense": offense_changes,
            }

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

        event = CPUDecisionEngine.assemble_event(
            game=game,
            batting_team=batting_team,
            fielding_team=fielding_team,
            batter=batter,
            pitcher=pitcher,
            prev_inning=prev_inning,
            prev_half=prev_half,
            action_label=action_label,
            result_key=result_key,
            message=message,
            runs_scored=runs_scored,
            roster_changes=roster_changes,
        )
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
