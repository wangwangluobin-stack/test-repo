"""数据清洗核心逻辑。

设计原则：
- 每个函数只做一件事，方便单测
- 所有清洗动作返回 (DataFrame, 日志dict)，日志用于写报告
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 缺失值填充策略
FILL_STRATEGIES = ("drop", "mean", "median", "mode", "zero")


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """删除完全重复的行。"""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, {"removed_duplicates": before - len(out)}


# 判定"这一列该不该转数值"的最低阈值：转换后有效值占比 >= 该比例才转。
# 0.5 意味着：一列里过半数值能转成数字，就认为它是数值列（夹杂少数脏字符串）。
# 城市/性别这类文本列可转比例通常接近 0，不会被误转。
COERCE_MIN_RATIO = 0.5


def coerce_numeric(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    min_ratio: float = COERCE_MIN_RATIO,
) -> tuple[pd.DataFrame, dict]:
    """把"看起来是数值"的列强制转成数值类型，无法转换的记为 NaN。

    关键：文本类别列（城市、性别、日期等）**不应该**被转成数值。
    判定规则：只有列中可成功转为数值的比例 >= min_ratio 时才转换，
    否则保持原样。这样 `income` 会被转（大部分是数字，夹杂少量字符串），
    而 `city` / `gender` 保持文本。
    """
    out = df.copy()
    targets = cols if cols is not None else list(out.columns)
    converted: dict[str, int] = {}
    skipped: list[str] = []

    for c in targets:
        if c not in out.columns:
            continue
        # 已经是纯数值类型就跳过
        if pd.api.types.is_numeric_dtype(out[c]):
            continue

        original_non_null = int(out[c].notna().sum())
        if original_non_null == 0:
            continue

        coerced = pd.to_numeric(out[c], errors="coerce")
        new_non_null = int(coerced.notna().sum())
        ratio = new_non_null / original_non_null

        # 有效数值占比太低 -> 这是文本列，不动它
        if ratio < min_ratio:
            skipped.append(c)
            continue

        n_bad = original_non_null - new_non_null
        if n_bad > 0:
            converted[c] = n_bad
        out[c] = coerced

    return out, {"coerced_to_numeric": converted, "skipped_non_numeric": skipped}


def fill_missing(df: pd.DataFrame, strategy: str = "median") -> tuple[pd.DataFrame, dict]:
    """按策略填充缺失值。

    - 数值列：mean / median / zero
    - 类别列：mode（众数）
    - drop：删除含缺失值的行
    """
    if strategy not in FILL_STRATEGIES:
        raise ValueError(f"未知填充策略: {strategy}，可选 {FILL_STRATEGIES}")

    out = df.copy()
    missing_before = {c: int(n) for c, n in out.isna().sum().items() if n > 0}

    if strategy == "drop":
        out = out.dropna().reset_index(drop=True)
        return out, {"missing_before": missing_before, "rows_dropped": len(df) - len(out)}

    for c in out.columns:
        if out[c].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(out[c]):
            if strategy == "mean":
                out[c] = out[c].fillna(out[c].mean())
            elif strategy == "median":
                out[c] = out[c].fillna(out[c].median())
            elif strategy == "zero":
                out[c] = out[c].fillna(0)
            else:  # mode 策略下数值列也回退到中位数
                out[c] = out[c].fillna(out[c].median())
        else:
            # 类别列统一用众数
            mode_vals = out[c].mode()
            if len(mode_vals) > 0:
                out[c] = out[c].fillna(mode_vals.iloc[0])

    return out, {"missing_before": missing_before, "rows_dropped": 0}


def clean_categorical(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """清理文本类别列：去首尾空格、统一转小写比对后保留首个写法。"""
    out = df.copy()
    targets = cols if cols is not None else [
        c for c in out.columns
        if not pd.api.types.is_numeric_dtype(out[c])
    ]
    changes: dict[str, int] = {}

    for c in targets:
        if c not in out.columns or pd.api.types.is_numeric_dtype(out[c]):
            continue
        before = out[c].astype(str)
        stripped = before.str.strip()
        # 用小写作为归一化 key，映射回出现次数最多的原始写法
        key = stripped.str.lower()
        mapping = (
            pd.DataFrame({"key": key, "raw": stripped})
            .drop_duplicates()
            .set_index("key")["raw"]
            .to_dict()
        )
        normalized = key.map(mapping)
        n_changed = int((normalized != before).sum())
        out[c] = normalized
        if n_changed:
            changes[c] = n_changed

    return out, {"categorical_normalized": changes}


def detect_outliers(df: pd.DataFrame, cols: list[str] | None = None, k: float = 1.5) -> dict:
    """用 IQR 方法检测异常值，只做统计不删除。

    Returns:
        {列名: {"count": n, "lower": lo, "upper": hi}}
    """
    targets = cols if cols is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    result: dict[str, dict] = {}

    for c in targets:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out > 0:
            result[c] = {
                "count": n_out,
                "lower": round(float(lo), 3),
                "upper": round(float(hi), 3),
            }
    return result


def remove_outliers(df: pd.DataFrame, cols: list[str] | None = None, k: float = 1.5) -> tuple[pd.DataFrame, dict]:
    """把 IQR 判定为异常值的行置为 NaN（不直接删除，保留行）。"""
    out = df.copy()
    info = detect_outliers(out, cols, k)
    for c, spec in info.items():
        mask = (out[c] < spec["lower"]) | (out[c] > spec["upper"])
        out.loc[mask, c] = np.nan
    return out, {"outliers_nan": {c: v["count"] for c, v in info.items()}}


def run_pipeline(
    df: pd.DataFrame,
    fill_strategy: str = "median",
    handle_outliers: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """完整清洗流水线，返回 (清洗后数据, 全流程日志)。"""
    log: dict = {}

    df, step = drop_duplicates(df)
    log.update(step)

    df, step = clean_categorical(df)
    log.update(step)

    df, step = coerce_numeric(df)
    log.update(step)

    if handle_outliers:
        df, step = remove_outliers(df)
        log.update(step)

    df, step = fill_missing(df, fill_strategy)
    log.update(step)

    log["final_shape"] = list(df.shape)
    return df, log
