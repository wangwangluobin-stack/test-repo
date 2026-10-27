import pandas as pd

from report import build_report


def test_report_contains_distribution_frequencies_and_strong_pair(tmp_path):
    df = pd.DataFrame({"x": range(10), "y": range(0, 20, 2), "city": ["北京"] * 8 + ["上海"] * 2})
    report = build_report(df, df, {}, tmp_path)
    assert report.index("清洗前 vs 清洗后") < report.index("## 自动发现")
    assert "近似对称" in report
    assert "| 北京 | 8 | 80.0% |" in report
    assert "| x | y | 1.000 | 10 |" in report
    assert "相关不代表因果" in report
    assert "## 1. 数据概览" in report
    assert "## 5. 结论与建议" in report


def test_report_with_only_categories_and_missing_values(tmp_path):
    df = pd.DataFrame({"city": ["北京", None, "北京", "上海"]})
    report = build_report(df, df, {}, tmp_path)
    assert "没有实数类型的数值列" in report
    assert "| （缺失） | 1 | 25.0% |" in report
    assert "没有满足 |r| ≥ 0.7" in report


def test_category_labels_cannot_break_new_markdown_tables(tmp_path):
    df = pd.DataFrame({"city": ["a|b\nsecond line"]})
    report = build_report(df, df, {}, tmp_path)
    assert "| a\\|b second line | 1 | 100.0% |" in report


def test_histogram_and_correlation_handle_infinite_values(tmp_path):
    from report import plot_corr, plot_hist

    df = pd.DataFrame({"x": [1.0, 2.0, float("inf")], "y": [3, 4, 5]})
    assert plot_hist(df, tmp_path).read_bytes().startswith(b'\x89PNG')
    assert plot_corr(df, tmp_path).read_bytes().startswith(b'\x89PNG')
