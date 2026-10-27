"""确保实际保存的报告顶部包含真实的清洗前后评分。"""

import pandas as pd
import pytest

from clean import run_pipeline
from quality import quality_score
from report import build_report, save_report


def test_saved_report_starts_with_quality_comparison(tmp_path):
    raw = pd.DataFrame({"value": [1.0, 1.0, None, 3.0]})
    cleaned, log = run_pipeline(raw)
    before, after = quality_score(raw), quality_score(cleaned)
    assert before["total_score"] == 85.0
    assert after["total_score"] == 100.0

    output = tmp_path / "report.md"
    save_report(build_report(raw, cleaned, log, tmp_path / "figures"), output)
    text = output.read_text(encoding="utf-8")
    intro = text.split("## 1. 数据概览")[0]
    assert text.startswith("# 数据清洗与探索性分析报告\n")
    assert "数据质量评分：清洗前 vs 清洗后" in intro
    assert "清洗前 85.00/100 → 清洗后 100.00/100（变化 +15.00 分）" in intro
    assert "| 完整度 | 35% | 75.00 | 100.00 | +25.00 |" in intro
    assert "| 唯一性 | 25% | 75.00 | 100.00 | +25.00 |" in intro
    assert "| 一致性 | 20% | 100.00 | 100.00 | +0.00 |" in intro
    assert "| 有效性 | 20% | 100.00 | 100.00 | +0.00 |" in intro
    assert all(reason in intro for reason in before["weight_reasons"].values())
    assert "不代表恢复了真实数据" in intro
    assert "## 5. 结论与建议" in text


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"x": [None, None]})])
def test_empty_report_explicitly_marks_no_data(df, tmp_path):
    text = build_report(df, df, {}, tmp_path)
    assert "清洗前 0.00/100 → 清洗后 0.00/100" in text
    assert "无可评估数据：清洗前、清洗后。" in text
    assert "清洗后数据完整度为 **0.00%**" in text


def test_report_preserves_negative_score_change(tmp_path):
    raw = pd.DataFrame({"value": [1, 2, 3]})
    degraded = pd.DataFrame({"value": [None, None, None]})
    text = build_report(raw, degraded, {}, tmp_path)
    assert "清洗前 100.00/100 → 清洗后 0.00/100（变化 -100.00 分）" in text
