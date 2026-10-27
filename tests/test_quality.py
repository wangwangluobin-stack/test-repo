"""独立核对评分公式、退化输入和输入保护。"""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quality import quality_score


@pytest.mark.parametrize("df", [
    pd.DataFrame(),
    pd.DataFrame(index=[0, 1]),
    pd.DataFrame(columns=["value"]),
    pd.DataFrame({"a": [np.nan, np.nan], "b": [None, None]}),
    pd.DataFrame({"a": pd.Series([pd.NA, pd.NA], dtype="Int64")}),
])
def test_empty_or_all_missing_scores_zero(df):
    result = quality_score(df)
    assert result["has_data"] is False
    assert result["total_score"] == 0.0
    assert result["scores"] == {
        "completeness": 0.0, "uniqueness": 0.0, "consistency": 0.0, "validity": 0.0,
    }
    json.dumps(result, allow_nan=False)


def test_perfect_data_scores_100_with_explained_weights():
    df = pd.DataFrame({"value": [-2, -1, 0, 1, 2], "city": list("abcde")})
    result = quality_score(df)
    assert result["has_data"] is True
    assert result["scores"] == {
        "completeness": 100.0, "uniqueness": 100.0,
        "consistency": 100.0, "validity": 100.0,
    }
    assert result["total_score"] == 100.0
    assert result["weights"] == {
        "completeness": 0.35, "uniqueness": 0.25, "consistency": 0.20, "validity": 0.20,
    }
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert set(result["weight_reasons"]) == set(result["weights"])
    assert all(result["weight_reasons"].values())


def test_single_numeric_column():
    result = quality_score(pd.DataFrame({"value": [1, 2, 3, 4]}))
    assert result["total_score"] == 100.0
    assert result["details"]["numeric_cells"] == 4
    assert result["details"]["iqr_checked_cells"] == 4
    assert result["details"]["outlier_cells"] == 0


def test_mixed_types_use_semantic_majority_and_non_missing_denominator():
    df = pd.DataFrame({
        "value": pd.Series([1, 2.0, "3", "bad", None], dtype=object),
        "city": list("abcde"),
    })
    result = quality_score(df)
    assert result["scores"] == {
        "completeness": 90.0, "uniqueness": 100.0,
        "consistency": 88.89, "validity": 100.0,
    }
    assert result["details"]["consistent_cells"] == 8
    assert result["details"]["non_missing_cells"] == 9
    assert result["total_score"] == 94.28


def test_duplicate_rows_reduce_uniqueness():
    result = quality_score(pd.DataFrame({"id": [1, 1, 2, 3]}))
    assert result["scores"]["uniqueness"] == 75.0
    assert result["details"]["duplicate_rows"] == 1
    assert result["total_score"] == 93.75


def test_missing_cells_only_reduce_completeness():
    result = quality_score(pd.DataFrame({"value": [1.0, np.nan, 3.0]}))
    assert result["scores"] == {
        "completeness": 66.67, "uniqueness": 100.0,
        "consistency": 100.0, "validity": 100.0,
    }
    assert result["total_score"] == 88.33


def test_iqr_outlier_reduces_validity():
    result = quality_score(pd.DataFrame({"value": [1, 2, 3, 4, 5, 1000]}))
    assert result["scores"]["validity"] == 83.33
    assert result["details"]["outlier_cells"] == 1
    assert result["details"]["iqr_checked_cells"] == 6
    assert result["total_score"] == 96.67


def test_infinities_are_invalid_and_excluded_from_quartiles():
    result = quality_score(pd.DataFrame({"value": [1, 2, 3, 4, np.inf, -np.inf]}))
    assert result["scores"]["validity"] == 66.67
    assert result["details"]["non_finite_cells"] == 2
    assert result["details"]["outlier_cells"] == 0
    assert result["details"]["iqr_checked_cells"] == 4


def test_small_sample_only_checks_finiteness():
    result = quality_score(pd.DataFrame({"value": [1, 1000, np.inf]}))
    assert result["scores"]["validity"] == 66.67
    assert result["details"]["iqr_checked_cells"] == 0
    assert result["details"]["outlier_cells"] == 0


def test_numeric_strings_participate_in_outlier_detection():
    df = pd.DataFrame({"value": ["1", "2", "3", "4", "5", "1000"]})
    result = quality_score(df)
    assert result["scores"]["validity"] == 83.33
    assert result["details"]["numeric_cells"] == 6


def test_text_only_does_not_claim_numeric_checks():
    result = quality_score(pd.DataFrame({"city": ["北京", "上海", "广州"]}))
    assert result["total_score"] == 100.0
    assert result["details"]["numeric_cells"] == 0
    assert result["details"]["iqr_checked_cells"] == 0


def test_boolean_is_separate_from_numeric():
    df = pd.DataFrame({"value": pd.Series([True, False, 1, 2], dtype=object)})
    result = quality_score(df)
    assert result["scores"]["consistency"] == 50.0
    assert result["details"]["numeric_cells"] == 2


def test_single_row_nullable_values_and_dates():
    df = pd.DataFrame({
        "number": pd.Series([1], dtype="Int64"),
        "flag": pd.Series([True], dtype="boolean"),
        "when": pd.to_datetime(["2024-01-01"]),
    })
    assert quality_score(df)["total_score"] == 100.0


def test_repeated_column_names_are_scored_by_position():
    df = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]], columns=["x", "x"])
    assert quality_score(df)["total_score"] == 100.0


def test_scoring_is_read_only_and_returns_independent_metadata():
    df = pd.DataFrame({"value": pd.Series([1, "bad", None], dtype=object)})
    original = df.copy(deep=True)
    result = quality_score(df)
    assert_frame_equal(df, original)
    result["weights"]["completeness"] = 0
    result["weight_reasons"].clear()
    again = quality_score(df)
    assert again["weights"]["completeness"] == 0.35
    assert again["weight_reasons"]["completeness"]
    assert 0 <= again["total_score"] <= 100
    assert all(0 <= score <= 100 for score in again["scores"].values())
    json.dumps(again, allow_nan=False)


def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        quality_score([1, 2, 3])
