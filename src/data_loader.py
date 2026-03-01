"""Data loading utilities for WHL 2025 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _validate_columns(df: pd.DataFrame, required_columns: Optional[Iterable[str]]) -> None:
    """Validate required columns exist in a DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: Columns that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """
    if not required_columns:
        return
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_csv(
    path: str | Path,
    *,
    dtype: Optional[Dict[str, Any]] = None,
    required_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a CSV file with basic validation.

    Args:
        path: Path to the CSV file.
        dtype: Optional pandas dtype mapping.
        required_columns: Optional list of required column names.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df = pd.read_csv(file_path, dtype=dtype)
    _validate_columns(df, required_columns)
    return df


def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int | None = 0,
    dtype: Optional[Dict[str, Any]] = None,
    required_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load an Excel file with basic validation.

    Args:
        path: Path to the Excel file.
        sheet_name: Excel sheet name or index.
        dtype: Optional pandas dtype mapping.
        required_columns: Optional list of required column names.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=dtype)
    _validate_columns(df, required_columns)
    return df