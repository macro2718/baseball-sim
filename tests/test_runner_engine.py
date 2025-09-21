import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

if "baseball_sim" not in sys.modules:
    fake_package = ModuleType("baseball_sim")
    fake_package.__path__ = [str(_PACKAGE_ROOT / "baseball_sim")]
    sys.modules["baseball_sim"] = fake_package

_UTILS_PATH = _PACKAGE_ROOT / "baseball_sim" / "gameplay" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("runner_utils", _UTILS_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RunnerEngine = _MODULE.RunnerEngine


class DummyGameState:
    def __init__(self):
        self.bases = [None, None, None]
        self.outs = 0

    def add_out(self):
        self.outs += 1


def test_apply_single_third_runner_stays_on_base():
    game_state = DummyGameState()
    third_runner = SimpleNamespace(speed=4.3)
    batter = SimpleNamespace(speed=4.3)
    game_state.bases[2] = third_runner

    engine = RunnerEngine(game_state)

    with patch("baseball_sim.gameplay.utils.random.random", side_effect=[0.99]):
        runs = engine.apply_single(batter)

    assert runs == 0
    assert game_state.bases[2] is third_runner
    assert game_state.bases[0] is batter
    assert game_state.outs == 0


def test_apply_single_third_runner_scores():
    game_state = DummyGameState()
    third_runner = SimpleNamespace(speed=4.3)
    batter = SimpleNamespace(speed=4.3)
    game_state.bases[2] = third_runner

    engine = RunnerEngine(game_state)

    with patch("baseball_sim.gameplay.utils.random.random", side_effect=[0.1, 0.1]):
        runs = engine.apply_single(batter)

    assert runs == 1
    assert game_state.bases[2] is None
    assert game_state.bases[0] is batter
    assert game_state.outs == 0
