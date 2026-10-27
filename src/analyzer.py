"""使用 pandas 自动探索数据，返回可序列化的统计与发现。"""

from __future__ import annotations

import math

import pandas as pd


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """布尔值是类别，复数不参与实数分布与相关性分析。"""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
            and not pd.api.types.is_complex_dtype(df[c])]


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([float("inf"), -float("inf")], float("nan")).dropna().astype(float)


def analyze(df: pd.DataFrame, correlation_threshold: float = 0.7,
            max_categories: int = 10) -> dict:
    """分析每个数值/类别列，找出 |Pearson r| 达阈值且至少有 3 对观测的特征对。

    分布仅使用样本偏度描述，不声称通过了正态性检验；不足 8 个有限观测不判断形态。
    类别频次包含缺失项，最多输出 max_categories 项，比例以全部行数为分母。
    不修改输入；列名需为唯一字符串（CSV 加载器的常规输出）。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("analyze 需要 pandas.DataFrame")
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("分析要求列名为唯一字符串")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("相关系数阈值必须在 (0, 1] 内")
    if not isinstance(max_categories, int) or max_categories < 1:
        raise ValueError("max_categories 必须为正整数")

    numeric = numeric_columns(df)
    result = {
        "overview": {"rows": len(df), "columns": len(df.columns),
                     "missing_cells": int(df.isna().sum().sum())},
        "numeric": {}, "categorical": {}, "correlations": [], "findings": [],
    }
    for column in df.columns:
        s = df[column]
        missing = int(s.isna().sum())
        if missing:
            result["findings"].append({
                "kind": "missing", "columns": [column],
                "message": f"{column} 有 {missing} 个缺失值。",
            })
        if column in numeric:
            finite = _finite(s)
            n = len(finite)
            skew = float(finite.skew()) if n >= 3 else None
            if skew is not None and not math.isfinite(skew):
                skew = None
            if n < 8:
                shape, description = "insufficient_data", "有限观测少于 8 个，不判断分布形态"
            elif finite.nunique() == 1:
                shape, description = "constant", "数值恒定"
            elif skew is None:
                shape, description = "insufficient_data", "无法稳定估计偏度"
            elif skew > 0.5:
                shape, description = "right_skewed", "右偏，较大值一侧尾部更长"
            elif skew < -0.5:
                shape, description = "left_skewed", "左偏，较小值一侧尾部更长"
            else:
                shape, description = "roughly_symmetric", "近似对称（仅偏度判断，非正态性检验）"
            outliers = 0
            if n >= 4:
                q1, q3 = finite.quantile([0.25, 0.75])
                outliers = int(((finite < q1 - 1.5 * (q3 - q1)) |
                                (finite > q3 + 1.5 * (q3 - q1))).sum())
            stats = {
                "count": int(s.notna().sum()), "missing": missing,
                "finite_count": n, "non_finite": int(s.notna().sum()) - n,
                "mean": float(finite.mean()) if n else None,
                "median": float(finite.median()) if n else None,
                "min": float(finite.min()) if n else None,
                "max": float(finite.max()) if n else None,
                "std": float(finite.std()) if n > 1 else None,
                "skewness": skew, "outliers": outliers,
                "distribution": {"kind": shape, "description": description},
            }
            result["numeric"][column] = stats
            result["findings"].append({
                "kind": "distribution", "columns": [column],
                "message": f"{column}：{description}；IQR 标记 {outliers} 个统计异常值。",
            })
        else:
            counts = s.value_counts(dropna=False)
            frequencies = [
                {"value": None if pd.isna(value) else str(value), "count": int(count),
                 "share": float(count / len(df))}
                for value, count in counts.head(max_categories).items()
            ]
            result["categorical"][column] = {
                "count": int(s.notna().sum()), "missing": missing,
                "unique": int(s.nunique()), "frequencies": frequencies,
                "truncated": len(counts) > max_categories,
            }
            if frequencies:
                top = frequencies[0]
                label = "缺失值" if top["value"] is None else top["value"]
                result["findings"].append({
                    "kind": "frequency", "columns": [column],
                    "message": f"{column} 最常见值为 {label}，占全部行的 {top['share']:.1%}。",
                })

    finite_df = df[numeric].replace([float("inf"), -float("inf")], float("nan"))
    for i, left in enumerate(numeric):
        for right in numeric[i + 1:]:
            pair = finite_df[[left, right]].dropna().astype(float)
            if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            coefficient = float(pair[left].corr(pair[right]))
            if math.isfinite(coefficient) and abs(coefficient) >= correlation_threshold:
                item = {"left": left, "right": right, "coefficient": coefficient,
                        "observations": len(pair)}
                result["correlations"].append(item)
                result["findings"].append({
                    "kind": "correlation", "columns": [left, right],
                    "message": f"{left} 与 {right} 强相关（r={coefficient:.3f}，"
                               f"{len(pair)} 对观测）；相关不代表因果。",
                })
    result["correlations"].sort(key=lambda item: abs(item["coefficient"]), reverse=True)
    return result
