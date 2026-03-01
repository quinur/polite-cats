"""Tests for data loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import load_csv


def test_load_csv_validates_columns(tmp_path):
    data = pd.DataFrame({"a": [1], "b": [2]})
    file_path = tmp_path / "sample.csv"
    data.to_csv(file_path, index=False)
    df = load_csv(file_path, required_columns=["a", "b"])
    assert list(df.columns) == ["a", "b"]


def test_load_csv_missing_columns(tmp_path):
    data = pd.DataFrame({"a": [1]})
    file_path = tmp_path / "sample.csv"
    data.to_csv(file_path, index=False)
    with pytest.raises(ValueError):
        load_csv(file_path, required_columns=["a", "b"])