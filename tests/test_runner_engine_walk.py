import importlib.machinery
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

# `baseball_sim` パッケージの import 時に heavy dependency を避けるため、
# テストから利用するサブモジュールのみを解決できる簡易パッケージを登録する。
package_root = Path(__file__).resolve().parents[1] / "baseball_sim"
if "baseball_sim" not in sys.modules:
    shim = ModuleType("baseball_sim")
    shim.__path__ = [str(package_root)]
    shim.__package__ = "baseball_sim"
    shim.__spec__ = importlib.machinery.ModuleSpec(
        name="baseball_sim", loader=None, is_package=True
    )
    sys.modules["baseball_sim"] = shim

from baseball_sim.gameplay.state import BaseRunners
from baseball_sim.gameplay.utils import RunnerEngine


def test_walk_with_second_and_third_does_not_score():
    runner_on_second = object()
    runner_on_third = object()
    batter = object()
    game_state = SimpleNamespace(bases=BaseRunners([None, runner_on_second, runner_on_third]))
    engine = RunnerEngine(game_state)

    runs = engine.apply_walk(batter)

    assert runs == 0
    assert game_state.bases[0] is batter
    assert game_state.bases[1] is runner_on_second
    assert game_state.bases[2] is runner_on_third
