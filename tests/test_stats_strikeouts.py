import sys
from pathlib import Path
from types import ModuleType


def _ensure_package_stub() -> None:
    """Mirror the stub logic used in other tests to avoid heavy optional deps."""
    package_name = "baseball_sim"
    if package_name in sys.modules:
        return

    package_path = Path(__file__).resolve().parents[1] / package_name
    stub = ModuleType(package_name)
    stub.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = stub


_ensure_package_stub()

from baseball_sim.data.player import Player
from baseball_sim.gameplay.statistics import StatsCalculator


def make_player():
    return Player(
        name="Test Player",
        eligible_positions=["CF"],
        k_pct=20,
        bb_pct=10,
        hard_pct=30,
        speed=4.3,
        gb_pct=45,
        fielding_skill=50,
    )


def test_record_strikeout_syncs_so_and_k_keys():
    player = make_player()

    StatsCalculator.record_strikeout(player.stats)

    assert player.stats["SO"] == 1
    assert player.stats["K"] == 1


def test_record_strikeout_handles_legacy_k_only_dict():
    stats = {"K": 2}

    StatsCalculator.record_strikeout(stats)

    assert stats["SO"] == 3
    assert stats["K"] == 3
