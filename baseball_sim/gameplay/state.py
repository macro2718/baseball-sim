"""Core data structures used to model the state of a baseball game."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, List

from baseball_sim.config import BASES_COUNT, OUTS_PER_INNING


class HalfInning(Enum):
    """Enumeration describing whether the current half inning is the top or bottom."""

    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class BaseRunners:
    """List-like wrapper that represents the occupants of each base."""

    _bases: List[Any] = field(default_factory=lambda: [None] * BASES_COUNT)

    def __post_init__(self) -> None:
        if len(self._bases) != BASES_COUNT:
            raise ValueError(f"BaseRunners requires exactly {BASES_COUNT} slots")

    def __getitem__(self, index: int) -> Any:
        return self._bases[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._bases[index] = value

    def __iter__(self) -> Iterator[Any]:
        return iter(self._bases)

    def __len__(self) -> int:
        return len(self._bases)

    def clear(self) -> None:
        """Remove all runners from the bases."""
        for i in range(BASES_COUNT):
            self._bases[i] = None

    def clone(self) -> "BaseRunners":
        """Return a shallow copy of the runner configuration."""
        return BaseRunners(self._bases.copy())

    def occupied_count(self) -> int:
        """Return the number of occupied bases."""
        return sum(1 for base in self._bases if base is not None)

    def is_empty(self) -> bool:
        """Return True when no runners are on base."""
        return self.occupied_count() == 0

    @classmethod
    def from_value(cls, value: Iterable[Any]) -> "BaseRunners":
        """Create a BaseRunners instance from an arbitrary iterable."""
        if isinstance(value, BaseRunners):
            return value.clone()
        bases = list(value)
        if len(bases) != BASES_COUNT:
            raise ValueError(f"Expected iterable with {BASES_COUNT} entries for bases")
        return cls(bases)


@dataclass
class HalfInningState:
    """Tracks the inning/half information and manages the out counter."""

    outs_per_inning: int = OUTS_PER_INNING
    inning: int = 1
    half: HalfInning = HalfInning.TOP
    outs: int = 0

    def register_out(self) -> bool:
        """Increment outs and return True if the half inning is complete."""
        self.outs += 1
        return self.outs >= self.outs_per_inning

    def reset_outs(self) -> None:
        """Reset the out counter for the next half inning."""
        self.outs = 0

    def move_to_bottom(self) -> None:
        """Transition from the top half to the bottom half of the inning."""
        self.half = HalfInning.BOTTOM

    def move_to_top_of_next_inning(self) -> None:
        """Advance to the next inning and start from the top half."""
        self.inning += 1
        self.half = HalfInning.TOP
        self.reset_outs()

    @property
    def is_top(self) -> bool:
        return self.half == HalfInning.TOP


@dataclass
class DefensiveStatus:
    """Represents the outcome of defensive alignment validation."""

    frozen: bool = False
    messages: List[str] = field(default_factory=list)

    def freeze(self, messages: Iterable[str]) -> None:
        self.frozen = True
        self.messages = list(messages)

    def warn(self, messages: Iterable[str]) -> None:
        self.frozen = False
        self.messages = list(messages)

    def clear(self) -> None:
        self.frozen = False
        self.messages.clear()


@dataclass
class Scoreboard:
    """Handles inning-by-inning and total scoring for both teams."""

    inning_scores: List[List[int]] = field(default_factory=lambda: [[], []])
    home_total: int = 0
    away_total: int = 0

    def open_new_inning(self) -> None:
        """Append blank scoring slots for a brand-new inning."""
        self.inning_scores[0].append(0)
        self.inning_scores[1].append(0)

    def ensure_inning(self, inning_index: int) -> None:
        """Make sure scoring slots exist for the provided inning index."""
        while len(self.inning_scores[0]) <= inning_index:
            self.open_new_inning()

    def add_runs(self, *, is_home: bool, inning_index: int, runs: int) -> None:
        """Record runs for the specified half inning."""
        if runs <= 0:
            return
        self.ensure_inning(inning_index)
        target = 1 if is_home else 0
        self.inning_scores[target][inning_index] += runs
        if is_home:
            self.home_total += runs
        else:
            self.away_total += runs

    @property
    def totals(self) -> dict:
        """Return aggregate scoring for both teams."""
        return {"home": self.home_total, "away": self.away_total}
