import sys
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
