"""Helpers to evaluate live in-game analytics via CPU simulations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from baseball_sim.config import GameResults
from baseball_sim.gameplay.cpu_strategy import (
    CPUOffenseDecision,
    CPUPlayType,
    plan_defensive_substitutions,
    plan_pinch_hit,
    plan_pinch_run,
    plan_pitcher_change,
    select_offense_play,
)
from baseball_sim.gameplay.substitutions import SubstitutionManager


@dataclass
class LiveAnalyticsResult:
    """Container for aggregated simulation metrics."""

    offense_key: Optional[str]
    samples: int
    expected_runs: float
    score_probability: float
    home_win_probability: float
    home_wins: int
    away_wins: int
    ties: int

    def as_payload(self) -> Dict[str, object]:
        return {
            "offense": self.offense_key,
            "samples": self.samples,
            "result": {
                "expected_runs": self.expected_runs,
                "score_probability": self.score_probability,
                "home_win_probability": self.home_win_probability,
                "home_wins": self.home_wins,
                "away_wins": self.away_wins,
                "ties": self.ties,
            },
        }


def simulate_live_analytics(game_state, samples: int = 100) -> LiveAnalyticsResult:
    """Run CPU simulations from ``game_state`` and aggregate half-inning metrics.

    The simulated sequences are truncated when the current offensive half-inning
    ends. ``expected_runs`` and ``score_probability`` therefore reflect the
    projected production for the remainder of the active half only. When the
    game continues beyond the half-inning, the simulation proceeds to a full
    conclusion so that win probabilities remain available.
    """

    if samples <= 0:
        samples = 1

    base_state = game_state
    if getattr(base_state, "game_ended", False):
        offense_key = _resolve_offense_key(base_state)
        winner = _determine_winner(base_state)
        home_win_probability = 1.0 if winner == "home" else 0.0
        if winner is None:
            home_win_probability = 0.5
        return LiveAnalyticsResult(
            offense_key=offense_key,
            samples=1,
            expected_runs=0.0,
            score_probability=0.0,
            home_win_probability=home_win_probability,
            home_wins=1 if winner == "home" else 0,
            away_wins=1 if winner == "away" else 0,
            ties=1 if winner is None else 0,
        )

    offense_key = _resolve_offense_key(base_state)
    base_home = getattr(base_state, "home_score", 0)
    base_away = getattr(base_state, "away_score", 0)

    half_inning_runs = 0.0
    half_inning_scoring_games = 0
    home_wins = 0
    away_wins = 0
    ties = 0

    for _ in range(samples):
        clone = deepcopy(base_state)
        starting_home = getattr(clone, "home_score", base_home)
        starting_away = getattr(clone, "away_score", base_away)

        _play_out_game(clone, stop_at_half_inning=True)

        mid_home = getattr(clone, "home_score", starting_home)
        mid_away = getattr(clone, "away_score", starting_away)

        if offense_key == "home":
            offense_runs = mid_home - starting_home
        elif offense_key == "away":
            offense_runs = mid_away - starting_away
        else:
            offense_runs = 0

        if offense_runs > 0:
            half_inning_scoring_games += 1
        half_inning_runs += offense_runs

        if not getattr(clone, "game_ended", False):
            _play_out_game(clone)

        final_home = getattr(clone, "home_score", base_home)
        final_away = getattr(clone, "away_score", base_away)

        winner = _determine_winner(clone)
        if winner == "home":
            home_wins += 1
        elif winner == "away":
            away_wins += 1
        else:
            ties += 1

    expected_runs = half_inning_runs / samples
    score_probability = half_inning_scoring_games / samples
    home_win_probability = (home_wins + 0.5 * ties) / samples

    return LiveAnalyticsResult(
        offense_key=offense_key,
        samples=samples,
        expected_runs=expected_runs,
        score_probability=score_probability,
        home_win_probability=home_win_probability,
        home_wins=home_wins,
        away_wins=away_wins,
        ties=ties,
    )


def _resolve_offense_key(game_state) -> Optional[str]:
    if game_state is None:
        return None
    if getattr(game_state, "batting_team", None) is getattr(game_state, "home_team", None):
        return "home"
    if getattr(game_state, "batting_team", None) is getattr(game_state, "away_team", None):
        return "away"
    return None


def _determine_winner(game_state) -> Optional[str]:
    home = getattr(game_state, "home_score", 0)
    away = getattr(game_state, "away_score", 0)
    if home > away:
        return "home"
    if away > home:
        return "away"
    return None


def _play_out_game(game_state, *, stop_at_half_inning: bool = False) -> None:
    """Advance ``game_state`` with CPU strategies.

    When ``stop_at_half_inning`` is ``True`` the progression stops once the
    offense finishes the current half-inning or the game ends.
    """

    offense_context: Dict[str, Tuple[Optional[int], int, bool, int, Tuple[int, int, int]]] = {}
    defense_context: Dict[str, Tuple[Optional[int], int, bool, int, Tuple[int, int, int]]] = {}

    target_inning = getattr(game_state, "inning", None)
    target_half_top = getattr(game_state, "is_top_inning", None)

    while not getattr(game_state, "game_ended", False):
        offense_team = getattr(game_state, "batting_team", None)
        defense_team = getattr(game_state, "fielding_team", None)
        if offense_team is None or defense_team is None:
            break

        offense_key = _resolve_offense_key(game_state)
        defense_key = "home" if offense_key == "away" else "away"

        if defense_key:
            _apply_defense_strategy(game_state, defense_team, defense_key, defense_context)
        if offense_key:
            _apply_offense_strategy(game_state, offense_team, offense_key, offense_context)

        substitution_manager = SubstitutionManager(offense_team)
        try:
            pinch_plan = plan_pinch_hit(game_state, offense_team, substitution_manager)
        except Exception:
            pinch_plan = None
        if pinch_plan:
            bench = substitution_manager.get_available_bench_players()
            if 0 <= pinch_plan.bench_index < len(bench):
                substitution_manager.execute_pinch_hit(pinch_plan.bench_index, pinch_plan.lineup_index)

        batter = getattr(offense_team, "current_batter", None)
        pitcher = getattr(defense_team, "current_pitcher", None)
        if batter is None or pitcher is None:
            break

        decision = select_offense_play(game_state, offense_team)
        play = decision.play if isinstance(decision, CPUOffenseDecision) else CPUPlayType.SWING

        prev_inning = game_state.inning
        prev_half_top = game_state.is_top_inning

        while True:
            if play is CPUPlayType.STEAL:
                if not game_state.can_steal():
                    play = CPUPlayType.SWING
                    continue
                steal_info = game_state.execute_steal()
                result_key = steal_info.get("result") if isinstance(steal_info, dict) else None
                if result_key == GameResults.STEAL_NOT_ALLOWED:
                    play = CPUPlayType.SWING
                    continue
                # Steal attempt completed; advance to next situation.
                break

            if play is CPUPlayType.SQUEEZE:
                if not game_state.can_squeeze():
                    play = CPUPlayType.SWING
                    continue
                game_state.execute_squeeze(batter, pitcher)
                pitcher.decrease_stamina()
                break

            if play is CPUPlayType.BUNT:
                if not game_state.can_bunt():
                    play = CPUPlayType.SWING
                    continue
                game_state.execute_bunt(batter, pitcher)
                pitcher.decrease_stamina()
                break

            # Default swing
            result = game_state.calculate_result(batter, pitcher)
            game_state.apply_result(result, batter)
            pitcher.decrease_stamina()
            break

        if getattr(game_state, "game_ended", False):
            break

        inning_changed = (prev_inning != game_state.inning) or (prev_half_top != game_state.is_top_inning)
        if play is not CPUPlayType.STEAL and not inning_changed:
            offense_team.next_batter()

        if stop_at_half_inning:
            if getattr(game_state, "game_ended", False):
                break
            if inning_changed:
                break
            if target_inning is not None and game_state.inning != target_inning:
                break
            if target_half_top is not None and game_state.is_top_inning != target_half_top:
                break


def _build_strategy_context(game_state) -> Tuple[Optional[int], int, bool, int, Tuple[int, int, int]]:
    last_sequence = None
    last_play = getattr(game_state, "last_play", None)
    if isinstance(last_play, dict):
        last_sequence = last_play.get("sequence")
        if last_sequence is not None:
            try:
                last_sequence = int(last_sequence)
            except (TypeError, ValueError):
                last_sequence = None
    base_signature = tuple(1 if base is not None else 0 for base in game_state.bases[:3])
    return (last_sequence, game_state.inning, game_state.is_top_inning, game_state.outs, base_signature)


def _apply_offense_strategy(game_state, offense_team, offense_key, context_store) -> None:
    context = _build_strategy_context(game_state)
    if context_store.get(offense_key) == context:
        return
    context_store[offense_key] = context

    substitution_manager = SubstitutionManager(offense_team)
    try:
        plan = plan_pinch_run(game_state, offense_team, substitution_manager)
    except Exception:
        plan = None
    if not plan:
        return

    bench = substitution_manager.get_available_bench_players()
    if not (0 <= plan.bench_index < len(bench)):
        return

    success, _ = substitution_manager.execute_defensive_substitution(plan.bench_index, plan.lineup_index)
    if not success:
        return

    lineup = getattr(offense_team, "lineup", [])
    if 0 <= plan.lineup_index < len(lineup) and plan.base_index < len(game_state.bases):
        game_state.bases[plan.base_index] = lineup[plan.lineup_index]


def _apply_defense_strategy(game_state, defense_team, defense_key, context_store) -> None:
    context = _build_strategy_context(game_state)
    if context_store.get(defense_key) == context:
        return
    context_store[defense_key] = context

    substitution_manager = SubstitutionManager(defense_team)
    plan = plan_pitcher_change(game_state, defense_team, substitution_manager)
    if plan:
        success, _ = substitution_manager.execute_pitcher_change(plan.pitcher_index)
        if success:
            _evaluate_alignment(game_state)
        return

    try:
        def_plans = plan_defensive_substitutions(game_state, defense_team, substitution_manager)
    except Exception:
        def_plans = []

    applied = False
    for dplan in def_plans:
        bench = substitution_manager.get_available_bench_players()
        bench_index = None
        if dplan.bench_player in bench:
            bench_index = bench.index(dplan.bench_player)
        else:
            for idx, player in enumerate(bench):
                if getattr(player, "name", None) == getattr(dplan, "incoming_name", None):
                    bench_index = idx
                    break
        if bench_index is None:
            continue
        success, _ = substitution_manager.execute_defensive_substitution(bench_index, dplan.lineup_index)
        if success:
            applied = True

    if applied:
        _evaluate_alignment(game_state)


def _evaluate_alignment(game_state) -> None:
    evaluator = getattr(game_state, "_evaluate_defensive_alignment", None)
    if callable(evaluator):
        evaluator()
