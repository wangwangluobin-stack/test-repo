import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from analyzer import analyze


@pytest.mark.parametrize(("values", "kind"), [
    (list(range(10)), "roughly_symmetric"),
    ([1] * 8 + [10, 100], "right_skewed"),
    ([-100, -10] + [-1] * 8, "left_skewed"),
    ([5] * 10, "constant"),
    ([1, 2, 3], "insufficient_data"),
])
def test_distribution_shapes(values, kind):
    result = analyze(pd.DataFrame({"value": values}))
    assert result["numeric"]["value"]["distribution"]["kind"] == kind
    assert result["numeric"]["value"]["mean"] == pytest.approx(sum(values) / len(values))
    assert result["findings"][0]["kind"] == "distribution"


def test_categories_include_nulls_and_report_truncation():
    result = analyze(pd.DataFrame({"city": ["北京", "北京", "上海", None],
                                  "flag": [True, True, False, False]}), max_categories=2)
    city = result["categorical"]["city"]
    assert city["unique"] == 2
    assert city["missing"] == 1
    assert city["frequencies"][0] == {"value": "北京", "count": 2, "share": 0.5}
    assert city["truncated"] is True
    full = analyze(pd.DataFrame({"city": ["北京", None]}))
    assert {item["value"] for item in full["categorical"]["city"]["frequencies"]} == {"北京", None}
    assert "flag" in result["categorical"]


def test_strong_pairs_are_unique_include_negative_and_skip_constants():
    df = pd.DataFrame({"a": range(10), "b": range(0, 20, 2),
                       "c": range(10, 0, -1), "constant": [1] * 10})
    pairs = analyze(df)["correlations"]
    assert {(p["left"], p["right"]) for p in pairs} == {("a", "b"), ("a", "c"), ("b", "c")}
    assert pairs[0]["observations"] == 10
    assert all(abs(p["coefficient"]) == pytest.approx(1.0) for p in pairs)
    assert any(p["coefficient"] < 0 for p in pairs)


def test_pairwise_missing_and_nonfinite_values():
    df = pd.DataFrame({"a": [1, 2, 3, np.inf, np.nan], "b": [2, 4, 6, 8, 10]})
    result = analyze(df)
    assert result["numeric"]["a"]["finite_count"] == 3
    assert result["numeric"]["a"]["non_finite"] == 1
    assert result["numeric"]["a"]["mean"] == 2
    assert result["correlations"][0]["observations"] == 3
    json.dumps(result, allow_nan=False)
    assert analyze(df.head(2))["correlations"] == []


def test_iqr_outliers_and_nullable_data():
    df = pd.DataFrame({"value": pd.Series([1, 2, 3, 4, 5, 1000, pd.NA], dtype="Int64")})
    result = analyze(df)
    assert result["numeric"]["value"]["outliers"] == 1
    assert result["numeric"]["value"]["missing"] == 1


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"value": [np.nan, np.nan]})])
def test_empty_data_serializes_without_nan(df):
    result = analyze(df)
    assert result["correlations"] == []
    json.dumps(result, allow_nan=False)


def test_analysis_does_not_modify_input():
    df = pd.DataFrame({"a": [1, np.inf, np.nan], "b": ["北京", None, "上海"]})
    original = df.copy(deep=True)
    analyze(df)
    assert_frame_equal(df, original)


@pytest.mark.parametrize("kwargs", [{"correlation_threshold": 0}, {"correlation_threshold": 1.1},
                                         {"max_categories": 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        analyze(pd.DataFrame(), **kwargs)


def test_duplicate_column_names_rejected():
    with pytest.raises(ValueError, match="唯一"):
        analyze(pd.DataFrame([[1, 2]], columns=["x", "x"]))
