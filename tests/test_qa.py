import copy
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from qa import answer_question


@pytest.fixture
def df():
    return pd.DataFrame({"age": [20, 30, 40, 50], "income": [100, 200, 300, 400],
                         "city": ["北京", "上海", "北京", None]})


@pytest.mark.parametrize(("question", "intent", "expected"), [
    ("数据有多少行？", "row_count", 4),
    ("有多少列", "column_count", 3),
    ("平均年龄", "mean", 35),
    ("收入的中位数", "median", 250),
    ("income sum", "sum", 1000),
    ("最小年龄", "min", 20),
    ("年龄最大值", "max", 50),
    ("age 标准差", "std", 12.909944487358056),
    ("城市有几种不同值", "unique", 2),
    ("重复多少行", "duplicates", 0),
])
def test_common_scalar_questions(df, question, intent, expected):
    result = answer_question(df, question)
    assert result["intent"] == intent
    assert result["data"] == pytest.approx(expected)


def test_overview_missing_frequency_correlation_top_and_grouping(df):
    assert answer_question(df, "数据概览")["data"]["columns"] == ["age", "income", "city"]
    assert answer_question(df, "缺失值统计")["data"] == {"age": 0, "income": 0, "city": 1}
    frequency = answer_question(df, "城市频次")
    assert frequency["intent"] == "frequency"
    assert frequency["data"][0] == {"value": "北京", "count": 2}
    correlation = answer_question(df, "年龄和收入的相关性")
    assert correlation["data"] == {"coefficient": pytest.approx(1), "observations": 4}
    assert "不代表因果" in correlation["answer"]
    top = answer_question(df, "收入最高的前2行")
    assert [row["income"] for row in top["data"]] == [400, 300]
    group = answer_question(df, "按城市分组计算收入平均值")
    assert group["intent"] == "groupby"
    assert group["data"] == [{"city": "北京", "income_mean": 200},
                             {"city": "上海", "income_mean": 200},
                             {"city": None, "income_mean": 400}]


def test_group_counts_take_precedence_over_row_count(df):
    result = answer_question(df, "按城市分组有多少人")
    assert result["intent"] == "groupby"
    assert result["data"][0] == {"city": "北京", "count": 2}


def test_followup_reuses_column_and_filters_then_reset(df):
    first = answer_question(df, "年龄大于30")
    assert first["data"]["rows"] == 2
    second = answer_question(df, "平均收入", first["context"])
    assert second["data"] == 350
    third = answer_question(df, "那中位数呢", second["context"])
    assert third["data"] == 350
    assert third["context"]["columns"] == ["income"]
    reset = answer_question(df, "重置筛选", third["context"])
    assert reset["context"]["filters"] == []
    assert answer_question(df, "平均收入", reset["context"])["data"] == 250


@pytest.mark.parametrize(("question", "expected"), [
    ("年龄大于等于30", 3), ("年龄小于等于30", 2), ("年龄 != 30", 3),
    ("城市等于北京", 2), ('城市等于"上海"', 1), ("城市等于北京有多少行", 2),
    ("年龄大于30有多少行", 2),
])
def test_filters(df, question, expected):
    result = answer_question(df, question)
    rows = result["data"] if result["intent"] == "row_count" else result["data"]["rows"]
    assert rows == expected


def test_filters_can_be_stacked_and_used_for_aggregation(df):
    first = answer_question(df, "年龄大于20")
    second = answer_question(df, "收入小于400", first["context"])
    assert second["data"]["rows"] == 2
    assert answer_question(df, "平均年龄", second["context"])["data"] == 35
    assert answer_question(df, "年龄大于30的平均收入")["data"] == 350


def test_empty_and_nonfinite_statistics_are_explicit():
    df = pd.DataFrame({"age": [np.nan, np.inf, -np.inf]})
    assert answer_question(df, "平均年龄")["data"] is None
    assert answer_question(df.iloc[:0], "总和 age")["data"] is None
    assert answer_question(df.iloc[:1], "age 标准差")["data"] is None


@pytest.mark.parametrize("question", [
    "", "天气怎么样", "平均值", "年龄收入平均值", "城市平均值", "不存在的列平均值",
    "age top 101", "age top 0", "年龄大于20小于40", "城市大于北京",
    "年龄 >", "__import__('os').system('echo hacked')",
])
def test_unsupported_or_ambiguous_questions_raise(df, question):
    with pytest.raises(ValueError):
        answer_question(df, question)


def test_unknown_column_does_not_silently_reuse_context(df):
    first = answer_question(df, "平均年龄")
    with pytest.raises(ValueError):
        answer_question(df, "salary平均值", first["context"])


def test_followup_can_change_column_and_keep_statistic(df):
    first = answer_question(df, "平均年龄")
    assert answer_question(df, "那收入呢", first["context"])["data"] == 250


def test_nullable_single_value_standard_deviation_is_unavailable():
    df = pd.DataFrame({"age": pd.Series([20], dtype="Int64")})
    assert answer_question(df, "年龄标准差")["data"] is None


def test_group_column_named_count_does_not_collide_with_result():
    df = pd.DataFrame({"count": ["a", "a", "b"]})
    assert answer_question(df, "按 count 分组计数")["data"] == [
        {"count": "a", "row_count": 2}, {"count": "b", "row_count": 1},
    ]


def test_inputs_are_not_modified_and_results_are_json_serializable(df):
    original = df.copy(deep=True)
    context = answer_question(df, "年龄大于20")["context"]
    old_context = copy.deepcopy(context)
    result = answer_question(df, "收入前2行", context)
    assert context == old_context
    assert_frame_equal(df, original)
    json.dumps(result, allow_nan=False)


def test_strong_correlations_without_explicit_columns(df):
    result = answer_question(df, "有哪些强相关特征")
    assert len(result["data"]) == 1
    assert {result["data"][0]["left"], result["data"][0]["right"]} == {"age", "income"}


def test_partial_ascii_column_names_do_not_match():
    df = pd.DataFrame({"id": [1], "paid": [20]})
    assert answer_question(df, "paid 平均值")["data"] == 20
    assert answer_question(df, "paid 平均值")["context"]["columns"] == ["paid"]


def test_comparison_characters_inside_quoted_text_are_values():
    df = pd.DataFrame({"city": ["a>b", "x"]})
    assert answer_question(df, '城市等于"a>b"')["data"]["rows"] == 1
