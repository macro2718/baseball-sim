"""Tests for RunnerEngine base-running logic during flyouts."""

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

package = ModuleType("baseball_sim")
package.__path__ = [str(ROOT / "baseball_sim")]
sys.modules.setdefault("baseball_sim", package)

utils_spec = importlib.util.spec_from_file_location(
    "gameplay_utils", ROOT / "baseball_sim" / "gameplay" / "utils.py"
)
gameplay_utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = gameplay_utils
assert utils_spec.loader is not None
utils_spec.loader.exec_module(gameplay_utils)
RunnerEngine = gameplay_utils.RunnerEngine

state_spec = importlib.util.spec_from_file_location(
    "gameplay_state", ROOT / "baseball_sim" / "gameplay" / "state.py"
)
gameplay_state = importlib.util.module_from_spec(state_spec)
sys.modules[state_spec.name] = gameplay_state
assert state_spec.loader is not None
state_spec.loader.exec_module(gameplay_state)
BaseRunners = gameplay_state.BaseRunners


class DummyGameState:
    """Minimal game state stub for runner engine testing."""

    def __init__(self, outs: int = 0):
        self.outs = outs
        self.bases = BaseRunners([None, None, None])
        pitcher = SimpleNamespace(pitching_stats={"IP": 0.0})
        self.fielding_team = SimpleNamespace(current_pitcher=pitcher)

    def add_out(self) -> None:
        self.outs += 1


def test_apply_flyout_sacrifice_fly_success(monkeypatch):
    game_state = DummyGameState(outs=0)
    third_runner = SimpleNamespace(speed=4.3)
    game_state.bases[2] = third_runner
    batter = SimpleNamespace(hard_pct=40)

    monkeypatch.setattr(random, "random", lambda: 0.1)

    engine = RunnerEngine(game_state)
    runs = engine.apply_flyout(batter)

    assert runs == 1
    assert game_state.bases[2] is None
    assert game_state.outs == 0


def test_apply_flyout_runner_thrown_out(monkeypatch):
    game_state = DummyGameState(outs=0)
    third_runner = SimpleNamespace(speed=4.3)
    game_state.bases[2] = third_runner
    batter = SimpleNamespace(hard_pct=35)

    monkeypatch.setattr(random, "random", lambda: 0.9)

    engine = RunnerEngine(game_state)
    runs = engine.apply_flyout(batter)

    assert runs == 0
    assert game_state.bases[2] is None
    assert game_state.outs == 1


def test_apply_flyout_other_runners_tag_up(monkeypatch):
    game_state = DummyGameState(outs=0)
    second_runner = SimpleNamespace(speed=4.3)
    first_runner = SimpleNamespace(speed=4.1)
    game_state.bases[1] = second_runner
    game_state.bases[0] = first_runner
    batter = SimpleNamespace(hard_pct=60)

    roll_sequence = iter([0.1, 0.2])
    monkeypatch.setattr(random, "random", lambda: next(roll_sequence))

    engine = RunnerEngine(game_state)
    runs = engine.apply_flyout(batter)

    assert runs == 0
    assert game_state.bases[2] is second_runner
    assert game_state.bases[1] is first_runner
    assert game_state.bases[0] is None
    assert game_state.outs == 0


def test_groundout_first_and_third_runner_holds(monkeypatch):
    game_state = DummyGameState(outs=0)
    first_runner = object()
    third_runner = object()
    batter = SimpleNamespace(speed=4.3)

    game_state.bases[0] = first_runner
    game_state.bases[2] = third_runner

    roll_sequence = iter([0.9, 0.8])  # ダブルプレー失敗 → 三塁走者は残塁
    monkeypatch.setattr(random, "random", lambda: next(roll_sequence))

    engine = RunnerEngine(game_state)
    runs, message = engine.apply_groundout(batter)

    assert runs == 0
    assert message == "Groundout, force at second."
    assert game_state.outs == 1
    assert game_state.bases[2] is third_runner  # 三塁走者が残留
    assert game_state.bases[1] is None
    assert game_state.bases[0] is batter


def test_groundout_first_and_third_runner_scores(monkeypatch):
    game_state = DummyGameState(outs=0)
    first_runner = object()
    third_runner = object()
    batter = SimpleNamespace(speed=4.3)

    game_state.bases[0] = first_runner
    game_state.bases[2] = third_runner

    roll_sequence = iter([0.9, 0.1])  # ダブルプレー失敗 → 三塁走者が生還
    monkeypatch.setattr(random, "random", lambda: next(roll_sequence))

    engine = RunnerEngine(game_state)
    runs, message = engine.apply_groundout(batter)

    assert runs == 1
    assert message == "Groundout, force at second. 1 run(s) scored!"
    assert game_state.outs == 1
    assert game_state.bases[2] is None
    assert game_state.bases[0] is batter


def test_groundout_bases_loaded_force_home(monkeypatch):
    game_state = DummyGameState(outs=0)
    first_runner = object()
    second_runner = object()
    third_runner = object()
    batter = SimpleNamespace(speed=4.3)

    game_state.bases[0] = first_runner
    game_state.bases[1] = second_runner
    game_state.bases[2] = third_runner

    roll_sequence = iter([0.9, 0.8])  # ダブルプレー失敗 → 本塁でフォースアウト
    monkeypatch.setattr(random, "random", lambda: next(roll_sequence))

    engine = RunnerEngine(game_state)
    runs, message = engine.apply_groundout(batter)

    assert runs == 0
    assert message == "Groundout, force at home."
    assert game_state.outs == 1
    assert game_state.bases[2] is second_runner  # 二塁走者が三塁へ
    assert game_state.bases[1] is first_runner  # 一塁走者が二塁へ
    assert game_state.bases[0] is batter
