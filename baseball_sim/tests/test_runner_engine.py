import sys
from itertools import chain, repeat
from pathlib import Path
import types

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PACKAGE_ROOT.parent))

if 'baseball_sim' in sys.modules:
    del sys.modules['baseball_sim']

baseball_sim_stub = types.ModuleType('baseball_sim')
baseball_sim_stub.__path__ = [str(PACKAGE_ROOT)]
sys.modules['baseball_sim'] = baseball_sim_stub

from baseball_sim.gameplay.state import BaseRunners
from baseball_sim.gameplay.utils import RunnerEngine


class DummyGameState:
    def __init__(self, outs: int = 0):
        self.outs = outs
        self.bases = BaseRunners()

    def add_out(self) -> None:
        self.outs += 1


def make_random_sequence(*values):
    sequence = chain(values, repeat(values[-1]))

    def _random():
        return next(sequence)

    return _random


def test_regular_groundout_runner_out_at_home(monkeypatch):
    game_state = DummyGameState(outs=0)
    game_state.bases[1] = "Runner2"
    game_state.bases[2] = "Runner3"

    engine = RunnerEngine(game_state)

    monkeypatch.setattr("baseball_sim.gameplay.utils.random.random", lambda: 0.95)

    runs, message = engine._handle_regular_groundout(batter=object())

    assert runs == 0
    assert message == "Groundout, runner out at home."
    assert game_state.outs == 2
    assert game_state.bases[2] == "Runner2"
    assert game_state.bases[1] is None


def test_apply_single_runner_thrown_out_attempting_third(monkeypatch):
    game_state = DummyGameState()
    runner = types.SimpleNamespace(speed=4.3)
    batter = types.SimpleNamespace(speed=4.3)
    game_state.bases[0] = runner

    engine = RunnerEngine(game_state)
    monkeypatch.setattr(
        "baseball_sim.gameplay.utils.random.random",
        make_random_sequence(0.05, 0.95),
    )

    runs = engine.apply_single(batter)

    assert runs == 0
    assert game_state.outs == 1
    assert game_state.bases[0] == batter
    assert game_state.bases[1] is None
    assert game_state.bases[2] is None


def test_apply_double_runner_scores_attempting_home(monkeypatch):
    game_state = DummyGameState()
    runner = types.SimpleNamespace(speed=4.3)
    batter = types.SimpleNamespace(speed=4.3)
    game_state.bases[0] = runner

    engine = RunnerEngine(game_state)
    monkeypatch.setattr(
        "baseball_sim.gameplay.utils.random.random",
        make_random_sequence(0.1, 0.05),
    )

    runs = engine.apply_double(batter)

    assert runs == 1
    assert game_state.outs == 0
    assert game_state.bases[0] is None
    assert game_state.bases[1] == batter
    assert game_state.bases[2] is None


def test_apply_triple_batter_scores_on_extra_base_attempt(monkeypatch):
    game_state = DummyGameState()
    batter = types.SimpleNamespace(speed=4.3)

    engine = RunnerEngine(game_state)
    monkeypatch.setattr(
        "baseball_sim.gameplay.utils.random.random",
        make_random_sequence(0.1, 0.1),
    )

    runs = engine.apply_triple(batter)

    assert runs == 1
    assert game_state.outs == 0
    assert all(base is None for base in game_state.bases)
