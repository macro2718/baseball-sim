"""Heuristic helpers that drive CPU decision making in CPU-vs-player games."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Tuple

from baseball_sim.config import GameResults, Positions


class CPUPlayType(str, Enum):
    """Enumerate the kinds of offensive actions the CPU can initiate."""

    SWING = "swing"
    BUNT = "bunt"
    STEAL = "steal"
    SQUEEZE = "squeeze"


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
    """刷新版: 投手交代判断ロジック

    方針:
      1. リリーフ資源(スタミナ>=20のRP)が残りイニング数より少ない場合は絶対に交代しない。
      2. 交代候補は RP を最優先。スタミナ>=20 の投手を優先し、同条件なら K%-BB% (k_pct - bb_pct) が高い順。
      3. 先発(SP) と リリーフ(RP) で交代指数(threshold) を変える。
      4. RP は原則 1 イニング単位運用: 新しいハーフイニング開始(=打者先頭 & outs==0)で前イニングから続投中なら指数閾値を低めに。
      5. 交代指数 = base + leverage + fatigue + score_component。
         ・current_stamina が低いほど上がる。
         ・ビハインド(負けている)かつ接戦は上がる。大量リードでは下げる。
      6. 閾値到達時のみ交代候補決定。

    戻り値: PitcherChangePlan か None
    """

    if not cpu_team or not hasattr(cpu_team, "current_pitcher"):
        return None

    current_pitcher = cpu_team.current_pitcher
    if current_pitcher is None:
        return None

    current_stamina = getattr(current_pitcher, "current_stamina", None)
    if current_stamina is None:
        return None

    inning = game_state.inning  # 1始まり想定
    outs = game_state.outs
    is_top = getattr(game_state, "is_top_inning", True)
    # 投手視点での点差: 自チーム(=守備側=cpu_team)の得点 - 相手得点
    # game_stateでは batting_team 目線だったので単純計算で置換
    score_margin = (game_state.home_score - game_state.away_score)
    if cpu_team is game_state.away_team:
        score_margin = -score_margin

    # 残りハーフイニング数(現在の守備ハーフ含めるかは交代消極性に影響) → 現在守備中ならこのハーフ含めて数える
    total_innings = 9  # 延長は現状考慮しない簡易版
    completed_full_innings = inning - 1
    remaining_full_innings_after_this = max(0, total_innings - inning)
    # 現在守備中のハーフ + 残りフルイニングの守備ハーフ数
    remaining_defensive_halves = 1 + remaining_full_innings_after_this

    # 利用可能RPを抽出
    available_pitchers = substitution_manager.get_available_pitchers()
    if not available_pitchers:
        return None

    def is_relief(p):
        return str(getattr(p, "pitcher_type", "RP") or "RP").upper() != "SP"

    ready_rps = [p for p in available_pitchers if is_relief(p) and (getattr(p, "current_stamina", getattr(p, "stamina", 0)) or 0) >= 20]
    ready_rp_count = len(ready_rps)

    # 資源不足なら絶対に交代しない (現投手がSPでもRPでも共通)
    if ready_rp_count < remaining_defensive_halves:
        return None

    pitcher_type = str(getattr(current_pitcher, "pitcher_type", "SP") or "SP").upper()

    # --- 交代指数計算 ---
    # スタミナ要素: 低いほど指数アップ。0..100 を想定し (100 - stamina)
    fatigue_component = (100 - float(current_stamina)) * 0.4  # 0..40

    # スコア要素: 負けているほど上昇 / 大量リードで減少
    # score_margin >0: リード, <0: ビハインド
    if score_margin < 0:
        # ビハインド: 最大 +20 まで ( -1~-2 接戦 +10, -3~-5 +15, -6以下 +20 )
        deficit = abs(score_margin)
        if deficit <= 2:
            score_component = 10
        elif deficit <= 5:
            score_component = 15
        else:
            score_component = 20
    elif score_margin > 0:
        lead = score_margin
        if lead <= 2:
            score_component = 2  # 接戦リード: ほぼ維持
        elif lead <= 5:
            score_component = -5
        else:
            score_component = -10
    else:
        score_component = 8  # 同点: ほどほどに上げる

    # レバレッジ (終盤接戦) ブースト
    high_leverage = (inning >= 7 and abs(score_margin) <= 2)
    leverage_component = 12 if high_leverage else 0

    # RP 1イニング原則: RP が続投して2イニング目に入るなら追加ブースト
    # 判定: outs==0 かつ そのイニングの先頭打者 (game_state.bases 全て空) で守備開始時点
    bases_empty = all(base is None for base in game_state.bases[:3])
    new_half_inning = (outs == 0 and bases_empty and getattr(game_state, "last_play", None) is not None)
    rp_one_inning_component = 0
    if pitcher_type != "SP" and new_half_inning:
        # リリーフをイニング跨ぎさせない方向
        rp_one_inning_component = 18

    base_component = 5  # ベースライン

    change_index = (
        base_component
        + fatigue_component
        + score_component
        + leverage_component
        + rp_one_inning_component
    )

    # --- 閾値設定 ---
    if pitcher_type == "SP":
        # 先発は基本我慢: 中盤までは高め、終盤や低スタミナで自然に上がりやすい調整
        threshold = 55
    else:
        # リリーフは原則短い: 継続条件を厳しく
        threshold = 45

    # イニング途中でのリリーフ交代は少し閾値を上げて無駄な継投を減らす
    if pitcher_type != "SP" and outs > 0:
        threshold += 5

    # 閾値未達なら交代しない
    if change_index < threshold:
        return None

    # --- 交代候補選定 ---
    def candidate_priority(p):
        p_type = str(getattr(p, "pitcher_type", "RP") or "RP").upper()
        is_rp = 0 if p_type != "SP" else 1  # RP優先
        p_stam = getattr(p, "current_stamina", getattr(p, "stamina", 0)) or 0
        high_stam_flag = 0 if p_stam >= 20 else 1  # 20以上を優先
        k_pct = float(getattr(p, "k_pct", 0.0) or 0.0)
        bb_pct = float(getattr(p, "bb_pct", 0.0) or 0.0)
        kbb = k_pct - bb_pct
        return (is_rp, high_stam_flag, -kbb, -p_stam)

    replacement_pool = list(available_pitchers)
    replacement_pool.sort(key=candidate_priority)
    replacement = replacement_pool[0]
    replacement_index = available_pitchers.index(replacement)

    reason = (
        f"change_index={change_index:.1f}>=threshold={threshold} (fatigue={fatigue_component:.1f}, score={score_component}, leverage={leverage_component}, one_inning={rp_one_inning_component})"
    )

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

    # --- Lightweight helpers for stronger CPU choices ---
    def _catcher_fielding_factor() -> float:
        """Return a 0.85..1.10 multiplier based on catcher fielding as a proxy for arm.

        - Strong catcher (fielding_skill ~ 90) -> ~0.85 (riskier to steal)
        - Average (60-70) -> ~1.0
        - Weak catcher (40-50) -> up to ~1.10 (easier to steal)
        """
        try:
            catcher = getattr(game_state.fielding_team, "defensive_positions", {}).get(
                Positions.CATCHER
            )
        except Exception:
            catcher = None
        skill = float(getattr(catcher, "fielding_skill", 70.0) or 70.0)
        # Map 40..90 -> 1.10..0.85 roughly linearly
        skill = max(30.0, min(95.0, skill))
        t = (skill - 60.0) / 30.0  # -1.0 .. ~1.17
        factor = 1.0 - 0.125 * t  # around 0.85..1.10
        return max(0.85, min(1.10, factor))

    def _estimate_steal_prob(runner) -> float:
        """Use game state's internal probability if available; otherwise approximate by speed."""
        if hasattr(game_state, "_steal_success_probability"):
            try:
                base = float(game_state._steal_success_probability(runner))
            except Exception:
                base = 0.7 * (float(getattr(runner, "speed", 100.0) or 100.0) / 100.0)
        else:
            base = 0.7 * (float(getattr(runner, "speed", 100.0) or 100.0) / 100.0)
        # Adjust based on catcher quality to avoid running on cannons
        return max(0.3, min(0.95, base * _catcher_fielding_factor()))

    def _is_high_leverage() -> bool:
        return inning >= 7 and abs(margin) <= 2

    # Smarter steal: use estimated success prob and game context thresholds
    if game_state.can_steal() and outs <= 1 and runners_present > 0:
        # Consider each eligible runner and take the best chance
        candidates = []
        if bases[0] is not None and bases[1] is None:
            candidates.append((0, bases[0]))
        if bases[1] is not None and bases[2] is None:
            candidates.append((1, bases[1]))
        # Double steal scenario: approximate by min of two probs
        if bases[0] is not None and bases[1] is not None and bases[2] is None:
            p1 = _estimate_steal_prob(bases[0])
            p2 = _estimate_steal_prob(bases[1])
            best_prob = min(p1, p2)
        else:
            best_prob = max((_estimate_steal_prob(r) for _, r in candidates), default=0.0)

        # Dynamic threshold: late and close -> be more aggressive if trailing/tied
        base_threshold = 0.73
        if _is_high_leverage():
            base_threshold -= 0.03 if margin <= 0 else 0.0  # slightly more aggressive when not leading
        if margin >= 2:
            base_threshold += 0.03  # protect lead

        if best_prob >= base_threshold:
            metadata = {
                "base_state": base_signature,
                "est_prob": round(best_prob, 3),
                "threshold": round(base_threshold, 3),
            }
            label = "好機と見て盗塁を仕掛けます。"
            return CPUOffenseDecision(CPUPlayType.STEAL, label=label, metadata=metadata)

    if (
        game_state.can_squeeze()
        and has_runner_on(2)
        and outs <= 1
        and (inning >= 6 or margin <= 0)
    ):
        third_speed = getattr(bases[2], "speed", 100.0) or 100.0
        batter_k_rate = getattr(batter, "k_pct", 25.0) or 25.0
        # Prefer squeeze with 0 outs; allow with 1 out only if very fast runner or high K batter
        out_penalty = 8.0 if outs == 1 else 0.0
        leverage_bonus = 6.0 if _is_high_leverage() and margin <= 0 else 0.0
        squeeze_score = (third_speed - out_penalty) + leverage_bonus + max(0.0, batter_k_rate - 24.0)
        if squeeze_score >= 106.0 or random.random() < 0.20:
            metadata = {
                "base_state": base_signature,
                "third_speed": third_speed,
                "k_pct": batter_k_rate,
                "score": round(squeeze_score, 1),
            }
            label = "勝負所、スクイズで確実に1点を取りに行きます。"
            return CPUOffenseDecision(CPUPlayType.SQUEEZE, label=label, metadata=metadata)

    if (
        game_state.can_bunt()
        and outs <= 1
        and (has_runner_on(0) or has_runner_on(1))
    ):
        # Late, close game and weak-contact, high-K batters bunt more often
        bunt_bias = float(getattr(batter, "hard_pct", 0.0) or 0.0)
        k_rate = float(getattr(batter, "k_pct", 25.0) or 25.0)
        gb_rate = float(getattr(batter, "gb_pct", 45.0) or 45.0)
        speed = float(getattr(batter, "speed", 100.0) or 100.0)

        late_close = inning >= 6 and margin <= 1
        runner_on_first = has_runner_on(0)

        bunt_score = 0.0
        if late_close:
            bunt_score += 12.0
        if runner_on_first and outs == 0:
            bunt_score += 10.0  # classic sac bunt spot
        # Penalize power bats, reward high K and ground-ball prone hitters
        bunt_score += max(0.0, (k_rate - 25.0) * 0.8)
        bunt_score += max(0.0, (50.0 - bunt_bias) * 0.6)
        bunt_score += max(0.0, (gb_rate - 45.0) * 0.3)  # avoid GDP risk by bunting
        # Very slow batter slightly more likely to bunt
        bunt_score += max(0.0, (105.0 - speed) * 0.2)

        threshold = 18.0  # tuned to remain conservative
        if bunt_score >= threshold and inning >= 6:
            metadata = {
                "base_state": base_signature,
                "bunt_score": round(bunt_score, 1),
                "k_pct": k_rate,
                "hard_pct": bunt_bias,
                "gb_pct": gb_rate,
            }
            label = "送りバントで走者を確実に進めます。"
            return CPUOffenseDecision(CPUPlayType.BUNT, label=label, metadata=metadata)

    return CPUOffenseDecision(CPUPlayType.SWING, label="通常打撃を選択します。", metadata={"base_state": base_signature})


def describe_steal_outcome(result_info: dict) -> str:
    """Return a short description string for steal outcomes."""

    result_key = result_info.get("result")
    message = str(result_info.get("message", ""))
    if result_key == GameResults.STOLEN_BASE:
        if "ダブルスチール" in message or "Double steal" in message:
            return "ダブルスチール成功"
        return "盗塁成功"
    if result_key == GameResults.CAUGHT_STEALING:
        return "盗塁失敗"
    return "盗塁判定"
