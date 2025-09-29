"""Shared data preparation helpers for prediction models."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_RATE_COLUMNS",
    "fetch_batting_data",
]

FEATURE_COLUMNS: List[str] = ["K%", "BB%", "Hard%", "GB%"]
TARGET_RATE_COLUMNS: List[str] = [
    "SO_rate",
    "BB_rate",
    "1B_rate",
    "2B_rate",
    "3B_rate",
    "HR_rate",
    "OTH_rate",
]

_RAW_REQUIRED_COLUMNS: Iterable[str] = [
    "K%",
    "BB%",
    "Hard%",
    "GB%",
    "PA",
    "SO",
    "BB",
    "1B",
    "2B",
    "3B",
    "HR",
]

_METADATA_COLUMNS: List[str] = ["year", "Name", "Team", "PA"]


def _ensure_single_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a frame that includes ``column`` (deriving it when necessary)."""

    if column in data:
        return data
    if column != "1B":
        raise KeyError(f"Unsupported derived column: {column}")

    singles = data["H"] - (data["2B"] + data["3B"] + data["HR"])
    return data.assign(**{column: singles})


def fetch_batting_data(
    start_year: int = 2020,
    end_year: int = 2024,
    min_pa: int = 100,
) -> pd.DataFrame:
    """Load batting statistics from ``pybaseball`` and compute rate columns."""

    from pybaseball import batting_stats

    frames = []
    for year in range(start_year, end_year + 1):
        df = batting_stats(year)
        df = df.copy()
        df["year"] = year
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["PA"] >= min_pa].reset_index(drop=True)
    data = _ensure_single_column(data, "1B")

    data = data.dropna(subset=_RAW_REQUIRED_COLUMNS).reset_index(drop=True)

    data = data.copy()
    others = data["PA"] - data[["SO", "BB", "1B", "2B", "3B", "HR"]].sum(axis=1)
    denominator = data["PA"].replace(0, 1)
    data["OTH_rate"] = others / denominator
    for column in ["SO", "BB", "1B", "2B", "3B", "HR"]:
        data[f"{column}_rate"] = data[column] / denominator

    selected_columns = (
        _METADATA_COLUMNS
        + [c for c in FEATURE_COLUMNS if c in data.columns]
        + [c for c in TARGET_RATE_COLUMNS if c in data.columns]
    )
    unique_columns = []
    seen = set()
    for column in selected_columns:
        if column in seen:
            continue
        unique_columns.append(column)
        seen.add(column)

    return data[unique_columns]
