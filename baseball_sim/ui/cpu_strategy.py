"""Heuristic helpers that drive CPU decision making in CPU-vs-player games."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Tuple

from baseball_sim.config import GameResults


class CPUPlayType(str, Enum):
    """Enumerate the kinds of offensive actions the CPU can initiate."""

    SWING = "swing"
    BUNT = "bunt"
    STEAL = "steal"


@dataclass(frozen=True)
class CPUOffenseDecision:
    """Structured description of what the CPU wants to do on offense."""

    play: CPUPlayType
    label: str = ""
    metadata: Dict[str, object] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class PitcherChangePlan:
    """Describe the CPU's desired pitcher change."""

    pitcher_index: int
    reason: str
    current_name: str
    replacement_name: str


@dataclass(frozen=True)
class PinchRunPlan:
    """Describe a pinch-run request for the CPU offense."""

    base_index: int
    bench_index: int
    lineup_index: int
    reason: str
    outgoing_name: str
    incoming_name: str


def _score_margin_for_offense(game_state, offense_team) -> int:
    """Return the score differential from the perspective of the batting team."""

    if offense_team is None:
        return 0
    home_margin = game_state.home_score - game_state.away_score
    return home_margin if offense_team is game_state.home_team else -home_margin


def _base_state_signature(bases: Sequence[object]) -> Tuple[int, int, int]:
    """Return a light-weight snapshot of base occupancy for heuristics."""

    return tuple(1 if runner is not None else 0 for runner in bases[:3])


def plan_pitcher_change(game_state, cpu_team, substitution_manager) -> Optional[PitcherChangePlan]:
    """Return a pitcher change plan when stamina or leverage requires it."""

    if not cpu_team or not hasattr(cpu_team, "current_pitcher"):
        return None

    current_pitcher = cpu_team.current_pitcher
    if current_pitcher is None:
        return None

    current_stamina = getattr(current_pitcher, "current_stamina", None)
    if current_stamina is None:
        return None

    inning = game_state.inning
    outs = game_state.outs
    margin = _score_margin_for_offense(game_state, game_state.batting_team)
    high_leverage = inning >= 7 and abs(margin) <= 2
    fatigue = current_stamina <= 12 or (inning >= 6 and current_stamina <= 25)
    late_inning_pressure = high_leverage and current_stamina <= 40
    long_inning_flag = outs == 2 and current_stamina <= 30

    if not (fatigue or late_inning_pressure or long_inning_flag):
        return None

    available = substitution_manager.get_available_pitchers()
    if not available:
        return None

    def pitcher_priority(pitcher) -> Tuple[int, float, float]:
        pitcher_type = str(getattr(pitcher, "pitcher_type", "RP") or "RP").upper()
        is_relief = 0 if pitcher_type != "SP" else 1
        stamina = getattr(pitcher, "current_stamina", None)
        if stamina is None:
            stamina = getattr(pitcher, "stamina", 0)
        strikeout = getattr(pitcher, "k_pct", 0.0) or 0.0
        return (is_relief, -float(stamina), -float(strikeout))

    replacement = sorted(available, key=pitcher_priority)[0]
    replacement_index = available.index(replacement)

    reason_bits = []
    if fatigue:
        reason_bits.append("スタミナ低下")
    if late_inning_pressure:
        reason_bits.append("終盤の接戦")
    if long_inning_flag and "終盤の接戦" not in reason_bits:
        reason_bits.append("イニング終盤")
    reason = " / ".join(reason_bits) if reason_bits else "状況判断"

    return PitcherChangePlan(
        pitcher_index=replacement_index,
        reason=reason,
        current_name=getattr(current_pitcher, "name", "current pitcher"),
        replacement_name=getattr(replacement, "name", "relief pitcher"),
    )


def plan_pinch_run(game_state, cpu_team, substitution_manager) -> Optional[PinchRunPlan]:
    """Return a pinch-running plan for a late-inning, close game situation."""

    if not cpu_team:
        return None

    inning = game_state.inning
    outs = game_state.outs
    margin = _score_margin_for_offense(game_state, cpu_team)

    if inning < 8 or outs >= 2:
        return None
    if margin < -2:
        return None

    bases = list(game_state.bases)
    slowest_speed = None
    target_base = None
    target_runner = None
    for base_index, runner in enumerate(bases[:3]):
        if runner is None:
            continue
        runner_speed = getattr(runner, "speed", 100.0) or 100.0
        if slowest_speed is None or runner_speed < slowest_speed:
            slowest_speed = runner_speed
            target_base = base_index
            target_runner = runner

    if target_runner is None or slowest_speed is None:
        return None

    available_bench = substitution_manager.get_available_bench_players()
    if not available_bench:
        return None

    runner_position = getattr(target_runner, "current_position", None)

    def eligible_candidates() -> Iterable[Tuple[int, object]]:
        for bench_index, bench_player in enumerate(available_bench):
            bench_speed = getattr(bench_player, "speed", 100.0) or 100.0
            can_play = True
            if runner_position and hasattr(bench_player, "can_play_position"):
                can_play = bench_player.can_play_position(runner_position)
            if not can_play:
                continue
            if bench_speed >= slowest_speed + 12:
                yield bench_index, bench_player

    best_option = None
    best_speed = None
    for bench_index, bench_player in eligible_candidates():
        bench_speed = getattr(bench_player, "speed", 100.0) or 100.0
        if best_option is None or bench_speed > best_speed:
            best_option = (bench_index, bench_player)
            best_speed = bench_speed

    if best_option is None:
        return None

    bench_index, bench_player = best_option
    lineup = getattr(cpu_team, "lineup", [])
    try:
        lineup_index = lineup.index(target_runner)
    except ValueError:
        return None

    reason = (
        "終盤の好機で走塁力を強化"
        if margin <= 0
        else "リード拡大を狙ってスピードアップ"
    )

    return PinchRunPlan(
        base_index=target_base,
        bench_index=bench_index,
        lineup_index=lineup_index,
        reason=reason,
        outgoing_name=getattr(target_runner, "name", "runner"),
        incoming_name=getattr(bench_player, "name", "pinch runner"),
    )


def select_offense_play(game_state, offense_team) -> CPUOffenseDecision:
    """Choose an offensive play using simple heuristics."""

    batter = getattr(offense_team, "current_batter", None)
    bases = list(game_state.bases)
    outs = game_state.outs
    inning = game_state.inning
    margin = _score_margin_for_offense(game_state, offense_team)

    def has_runner_on(base_index: int) -> bool:
        return base_index < len(bases) and bases[base_index] is not None

    runners_present = sum(1 for runner in bases[:3] if runner is not None)
    base_signature = _base_state_signature(bases)

    if game_state.can_steal() and outs <= 1 and runners_present > 0:
        fastest_speed = 0.0
        for runner in bases[:3]:
            if runner is None:
                continue
            fastest_speed = max(fastest_speed, getattr(runner, "speed", 100.0) or 100.0)
        steal_threshold = 110.0
        leverage_bonus = 5.0 if inning >= 7 and margin <= 1 else 0.0
        if fastest_speed >= steal_threshold - leverage_bonus:
            metadata = {
                "base_state": base_signature,
                "fastest_speed": fastest_speed,
            }
            label = "好走者を活かして盗塁を狙います。"
            return CPUOffenseDecision(CPUPlayType.STEAL, label=label, metadata=metadata)

    if (
        game_state.can_bunt()
        and outs <= 1
        and inning >= 6
        and margin <= 1
        and (has_runner_on(0) or has_runner_on(1))
    ):
        bunt_bias = getattr(batter, "hard_pct", 0.0) or 0.0
        strikeout_rate = getattr(batter, "k_pct", 25.0) or 25.0
        risk_factor = random.random()
        if bunt_bias < 35.0 or strikeout_rate >= 27.5 or risk_factor < 0.4:
            metadata = {
                "base_state": base_signature,
                "k_pct": strikeout_rate,
                "hard_pct": bunt_bias,
            }
            label = "送りバントで確実に進塁を狙います。"
            return CPUOffenseDecision(CPUPlayType.BUNT, label=label, metadata=metadata)

    return CPUOffenseDecision(CPUPlayType.SWING, label="通常打撃を選択します。", metadata={"base_state": base_signature})


def describe_steal_outcome(result_info: dict) -> str:
    """Return a short description string for steal outcomes."""

    result_key = result_info.get("result")
    message = str(result_info.get("message", ""))
    if result_key == GameResults.STOLEN_BASE:
        if "ダブルスチール" in message:
            return "ダブルスチール成功"
        return "盗塁成功"
    if result_key == GameResults.CAUGHT_STEALING:
        return "盗塁失敗"
    return "盗塁判定"

