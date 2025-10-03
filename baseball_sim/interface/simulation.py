from typing import Dict, Iterable, Mapping, Optional, Sequence

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
from baseball_sim.interface.game_management import reset_team_and_players, update_statistics
from baseball_sim.interface.league_setup import (
    TeamContext,
    build_team_contexts,
    generate_round_robin_pairs,
    initialize_league_results,
)

setup_project_environment()
PATHS = get_project_paths()

LOGGER = root_logger.getChild("interface.simulation")


class LeagueSimulator:
    """Coordinate multi-game or multi-team simulations."""

    def __init__(self, *, progress_callback=None, message_callback=None) -> None:
        self.progress_callback = progress_callback
        self.message_callback = message_callback

    def run_league(
        self,
        *,
        team_datas: Sequence[Mapping[str, object]],
        games_per_card: int,
        cards_per_opponent: int,
        role_assignment: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, object]:
        """Simulate a full league schedule and return aggregated results."""

        if games_per_card <= 0:
            raise ValueError("games_per_card must be greater than zero.")
        if cards_per_opponent <= 0:
            raise ValueError("cards_per_opponent must be greater than zero.")

        contexts = build_team_contexts(team_datas)
        if len(contexts) % 2 != 0:
            raise ValueError("League simulation requires an even number of teams.")

        results = initialize_league_results(contexts, role_assignment=role_assignment)

        rounds = generate_round_robin_pairs(len(contexts))
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
                    if self.message_callback:
                        self.message_callback(
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

                    if self.progress_callback:
                        self.progress_callback(day_number, total_days)

                    day_number += 1

        results["meta"]["completed_games"] = games_played
        results["meta"]["completed_days"] = day_number - 1

        return results

    def run_series(
        self,
        *,
        num_games: int,
        home_team_data: Optional[Mapping[str, object]] = None,
        away_team_data: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        """Simulate a series between two teams, loading defaults when required."""

        from baseball_sim.data.loader import DataLoader

        team_data_path = PATHS.get_teams_data_path()
        raw_team_data = DataLoader.load_json_data(team_data_path)
        if not isinstance(raw_team_data, Mapping):
            raise ValueError("Team data file is invalid or missing required structure.")

        base_home = home_team_data or raw_team_data.get("home_team")
        base_away = away_team_data or raw_team_data.get("away_team")
        if not isinstance(base_home, Mapping) or not isinstance(base_away, Mapping):
            raise ValueError("Home/Away team data could not be loaded.")

        games_per_card = int(num_games) if num_games else 0
        if games_per_card <= 0:
            raise ValueError("リーグシミュレーションには1試合以上の設定が必要です。")

        return self.run_league(
            team_datas=[dict(base_home), dict(base_away)],
            games_per_card=games_per_card,
            cards_per_opponent=1,
            role_assignment={"home": 0, "away": 1},
        )


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

    simulator = LeagueSimulator(
        progress_callback=progress_callback, message_callback=message_callback
    )

    # リーグ構成を決定
    if league_options is not None:
        team_datas = list(league_options.get("teams", []))
        games_per_card = int(league_options.get("games_per_card", 0) or 0)
        cards_per_opponent = int(league_options.get("cards_per_opponent", 0) or 0)
        role_assignment = league_options.get("role_assignment")

        if games_per_card <= 0 or cards_per_opponent <= 0:
            raise ValueError("リーグシミュレーションには1試合以上の設定が必要です。")

        results = simulator.run_league(
            team_datas=team_datas,
            games_per_card=games_per_card,
            cards_per_opponent=cards_per_opponent,
            role_assignment=role_assignment,
        )
    else:
        results = simulator.run_series(
            num_games=int(num_games) if num_games is not None else 0,
            home_team_data=home_team_data,
            away_team_data=away_team_data,
        )

    completion_message = "Simulation complete."

    if message_callback:
        message_callback(completion_message)
    else:
        print(completion_message)

    return results

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


__all__ = ["LeagueSimulator", "simulate_games", "simulate_single_game"]
