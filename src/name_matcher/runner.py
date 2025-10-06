"""Convenience helpers for running the Name Matcher end-to-end."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .pipeline import NameMatcher, NameMatcherConfig, NameMatcherResult


def match_file(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[NameMatcherConfig] = None,
) -> NameMatcherResult | None:
    """Run the full workflow on `input_path` and write the annotated results."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    try:
        dataframe = _load_dataframe(input_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_path}'. Please update the INPUT_FILE path in the configuration.")
        return None
    except ValueError:
        print(f"ERROR: Unsupported file format for '{input_path}'. Please provide a CSV or Excel file.")
        return None

    config = config or NameMatcherConfig()
    if config.name_column not in dataframe.columns:
        print(
            f"ERROR: Column '{config.name_column}' not found in '{input_path}'. Please check NAME_COLUMN in the configuration."
        )
        return None

    matcher = NameMatcher(config)
    try:
        return matcher.cluster(dataframe, output_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return None


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError("unsupported format")
