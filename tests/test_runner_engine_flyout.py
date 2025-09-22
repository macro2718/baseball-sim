from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys

import pytest


def _ensure_package_stub():
    """Ensure the lightweight package stub exists to avoid heavy imports."""
    package_name = "baseball_sim"
    if package_name in sys.modules:
        return

    package_path = Path(__file__).resolve().parents[1] / package_name
    stub = ModuleType(package_name)
    stub.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = stub


_ensure_package_stub()

from baseball_sim.gameplay.state import BaseRunners
from baseball_sim.gameplay.utils import RunnerEngine


class DummyGameState:
    def __init__(self, outs=0):
        self.outs = outs
        self.bases = BaseRunners()
        self.add_out_calls = 0

    def add_out(self):
        self.outs += 1
        self.add_out_calls += 1


def set_random_sequence(monkeypatch, sequence):
    iterator = iter(sequence)

    def fake_random():
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover - defensive programming
            raise AssertionError("random.random called more times than expected") from exc

    monkeypatch.setattr("baseball_sim.gameplay.utils.random.random", fake_random)


@pytest.fixture
def batter():
    return SimpleNamespace(hard_pct=35, speed=4.3)


def test_flyout_tagups_success(monkeypatch, batter):
    game_state = DummyGameState()
    engine = RunnerEngine(game_state)

    second_runner = SimpleNamespace(speed=4.3)
    first_runner = SimpleNamespace(speed=4.3)
    game_state.bases[1] = second_runner
    game_state.bases[0] = first_runner

    set_random_sequence(monkeypatch, [0.0, 0.0, 0.0, 0.0])

    runs = engine.apply_outfield_flyout(batter)

    assert runs == 0
    assert game_state.outs == 0
    assert game_state.bases[2] is second_runner
    assert game_state.bases[1] is first_runner
    assert game_state.bases[0] is None


def test_flyout_tagup_runner_thrown_out(monkeypatch, batter):
    game_state = DummyGameState()
    engine = RunnerEngine(game_state)

    second_runner = SimpleNamespace(speed=4.3)
    game_state.bases[1] = second_runner

    set_random_sequence(monkeypatch, [0.0, 0.99])

    runs = engine.apply_outfield_flyout(batter)

    assert runs == 0
    assert game_state.outs == 1
    assert game_state.bases[1] is None


def test_flyout_double_play_leave_early(monkeypatch, batter):
    game_state = DummyGameState()
    engine = RunnerEngine(game_state)

    second_runner = SimpleNamespace(speed=4.3)
    first_runner = SimpleNamespace(speed=4.3)
    game_state.bases[1] = second_runner
    game_state.bases[0] = first_runner

    set_random_sequence(monkeypatch, [0.99, 0.0, 0.99, 0.0])

    runs = engine.apply_outfield_flyout(batter)

    assert runs == 0
    assert game_state.outs == 2
    assert game_state.bases[1] is None
    assert game_state.bases[0] is None
    assert game_state.add_out_calls == 2


def test_infield_flyout_no_advancement(monkeypatch, batter):
    game_state = DummyGameState()
    engine = RunnerEngine(game_state)

    first_runner = SimpleNamespace(speed=4.3)
    game_state.bases[0] = first_runner

    set_random_sequence(monkeypatch, [0.5])

    runs = engine.apply_infield_flyout(batter)

    assert runs == 0
    assert game_state.outs == 0
    assert game_state.bases[0] is first_runner
    assert game_state.add_out_calls == 0


def test_infield_flyout_double_off_lead_runner(monkeypatch, batter):
    game_state = DummyGameState()
    engine = RunnerEngine(game_state)

    first_runner = SimpleNamespace(speed=4.3)
    game_state.bases[0] = first_runner

    set_random_sequence(monkeypatch, [0.01])

    runs = engine.apply_infield_flyout(batter)

    assert runs == 0
    assert game_state.outs == 1
    assert game_state.bases[0] is None
    assert game_state.add_out_calls == 1
