"""clean.py 的单元测试。

运行:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 让测试能 import src 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clean import (  # noqa: E402
    drop_duplicates,
    coerce_numeric,
    fill_missing,
    clean_categorical,
    detect_outliers,
    remove_outliers,
    run_pipeline,
)


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, log = drop_duplicates(df)
    assert len(out) == 2
    assert log["removed_duplicates"] == 1


def test_coerce_numeric():
    df = pd.DataFrame({"a": ["1", "2", "abc", "4"]})
    out, log = coerce_numeric(df, ["a"])
    assert pd.api.types.is_numeric_dtype(out["a"])
    assert out["a"].isna().sum() == 1
    assert log["coerced_to_numeric"]["a"] == 1


def test_fill_missing_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out, log = fill_missing(df, "median")
    assert out["a"].isna().sum() == 0
    assert out.loc[1, "a"] == 2.0


def test_fill_missing_drop():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out, log = fill_missing(df, "drop")
    assert len(out) == 2
    assert log["rows_dropped"] == 1


def test_fill_missing_invalid_strategy():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError):
        fill_missing(df, "not_a_strategy")


def test_clean_categorical():
    df = pd.DataFrame({"city": ["北京", " 北京 ", "北京"]})
    out, log = clean_categorical(df, ["city"])
    assert out["city"].nunique() == 1
    assert log["categorical_normalized"]["city"] == 1


def test_detect_outliers():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})
    info = detect_outliers(df, ["a"])
    assert "a" in info
    assert info["a"]["count"] >= 1


def test_remove_outliers():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    out, log = remove_outliers(df, ["a"])
    assert out["a"].isna().sum() >= 1
    assert log["outliers_nan"]["a"] >= 1


def test_run_pipeline_end_to_end():
    df = pd.DataFrame({
        "id": [1, 2, 2],
        "val": [10.0, np.nan, 1000.0],
        "cat": ["A", " A ", "A"],
    })
    out, log = run_pipeline(df, fill_strategy="median")
    assert out.isna().sum().sum() == 0
    assert "final_shape" in log
