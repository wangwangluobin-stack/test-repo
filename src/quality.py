"""不修改输入的、无需业务 schema 的数据质量评分。

缺失只按 pandas 的 NA 定义判断；数值字符串与实数归为同一类型，布尔值单独分类。
有效性只检查可解析实数的有限性与每列 1.5 × IQR 异常值，不推断业务取值范围。
少于四个有限数值时不做 IQR 检测，无数值时有效性为 100（表示未发现数值问题）。
空表、没有列或全为 NA 时，全部得分为 0，并用 has_data=False 标明无可评估数据。
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from numbers import Real

import numpy as np
import pandas as pd


WEIGHTS = {
    "completeness": 0.35,
    "uniqueness": 0.25,
    "consistency": 0.20,
    "validity": 0.20,
}

LABELS = {
    "completeness": "完整度",
    "uniqueness": "唯一性",
    "consistency": "一致性",
    "validity": "有效性",
}

WEIGHT_REASONS = {
    "completeness": "35%：缺失值直接减少可用信息，且会影响后续统计，因此优先级最高。",
    "uniqueness": "25%：重复行会重复计权、扭曲统计，但是否应当去重仍取决于业务。",
    "consistency": "20%：列内类型混杂会影响解析和计算，但无 schema 的类型推断有局限。",
    "validity": "20%：非有限值和 IQR 异常值影响分析，但统计异常可能是真实观测，避免过度惩罚。",
}


def _kind_and_number(value: object) -> tuple[str, float | None]:
    """归并兼容的标量类型；数值字符串参与有效性检查。"""
    if isinstance(value, (bool, np.bool_)):
        return "boolean", None
    if isinstance(value, (Real, Decimal)):
        try:
            return "numeric", float(value)
        except OverflowError:
            return "numeric", float("inf")
    if isinstance(value, str):
        try:
            return "numeric", float(value)
        except ValueError:
            return "text", None
    if isinstance(value, (date, datetime, np.datetime64)):
        return "datetime", None
    return type(value).__name__, None


def quality_score(df: pd.DataFrame) -> dict:
    """返回四项 0–100 分、总分、权重、权重理由及可核对的计数。

    完整度 = 非 NA 单元格 / 全部单元格。
    唯一性 = 去除完全重复行后的行数 / 全部行数（不检查主键）。
    一致性 = 各列主导语义类型的非 NA 单元格数之和 / 非 NA 单元格数。
    有效性 = 有限且未被 IQR 标记的数值 / 可解析的数值。
    缺失值只在完整度中扣分；无数值但有其他数据时，有效性记 100。
    各项保留两位小数，总分为这些公开得分的加权和，亦保留两位小数。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("quality_score 需要 pandas.DataFrame")

    non_missing = int(df.notna().to_numpy().sum())
    details = {
        "rows": len(df),
        "columns": df.shape[1],
        "cells": int(df.size),
        "non_missing_cells": non_missing,
        "missing_cells": int(df.size) - non_missing,
        "duplicate_rows": int(df.duplicated().sum()) if len(df) and df.shape[1] else 0,
        "consistent_cells": 0,
        "numeric_cells": 0,
        "non_finite_cells": 0,
        "outlier_cells": 0,
        "iqr_checked_cells": 0,
    }
    scores = dict.fromkeys(WEIGHTS, 0.0)

    if non_missing:
        # 按位置逐列处理，兼容重名列；类型占比按非 NA 单元格数量加权。
        for _, column in df.items():
            kinds: Counter = Counter()
            numbers: list[float] = []
            for value in column.dropna():
                kind, number = _kind_and_number(value)
                kinds[kind] += 1
                if number is not None:
                    numbers.append(number)
            details["consistent_cells"] += max(kinds.values(), default=0)

            numeric = np.asarray(numbers, dtype=float)
            finite = numeric[np.isfinite(numeric)]
            details["numeric_cells"] += len(numeric)
            details["non_finite_cells"] += len(numeric) - len(finite)
            if len(finite) >= 4:
                q1, q3 = np.quantile(finite, [0.25, 0.75])
                lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                details["outlier_cells"] += int(((finite < lower) | (finite > upper)).sum())
                details["iqr_checked_cells"] += len(finite)

        invalid = details["non_finite_cells"] + details["outlier_cells"]
        numeric_count = details["numeric_cells"]
        scores = {
            "completeness": round(100 * non_missing / df.size, 2),
            "uniqueness": round(100 * (len(df) - details["duplicate_rows"]) / len(df), 2),
            "consistency": round(100 * details["consistent_cells"] / non_missing, 2),
            "validity": round(100 * (numeric_count - invalid) / numeric_count, 2)
            if numeric_count else 100.0,
        }

    return {
        "total_score": round(sum(scores[key] * weight for key, weight in WEIGHTS.items()), 2),
        "scores": scores,
        "weights": WEIGHTS.copy(),
        "weight_reasons": WEIGHT_REASONS.copy(),
        "has_data": bool(non_missing),
        "details": details,
    }
