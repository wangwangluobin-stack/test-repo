"""数据读取模块。

负责把 CSV 安全地读进 pandas，并尽量自动处理编码问题。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 常见编码，按尝试顺序排列
ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1")


def load_csv(path: str | Path) -> pd.DataFrame:
    """读取 CSV，自动尝试常见编码。

    Args:
        path: CSV 文件路径。

    Returns:
        读取后的 DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 所有编码都失败。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(p, encoding=enc)
            if df.shape[0] == 0:
                continue
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            continue

    raise ValueError(f"无法解析 CSV 文件: {p}，最后一次错误: {last_err}")


def overview(df: pd.DataFrame) -> dict:
    """返回数据概览信息（用于报告）。"""
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }
