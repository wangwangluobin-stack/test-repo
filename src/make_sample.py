"""示例脏数据生成器。

生成一份故意弄脏的 CSV，用于测试清洗流程：
- 缺失值
- 重复行
- 类型混乱（数字列里混入字符串）
- 异常值（极端大/小的数值）
- 空白字符与大小写不一致的类别列
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "sample.csv"

RANDOM_SEED = 42
N_ROWS = 200


def make_sample(seed: int = RANDOM_SEED, n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cities = ["北京", "上海", "广州", "深圳", "杭州"]
    genders = ["男", "女"]

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "age": rng.integers(18, 60, size=n).astype(float),
            "income": rng.normal(12000, 4000, size=n).round(2),
            "score": rng.normal(75, 10, size=n).round(1),
            "city": rng.choice(cities, size=n),
            "gender": rng.choice(genders, size=n),
            "signup_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        }
    )

    # --- 人为注入脏数据 ---

    # 1) 缺失值：age / income / score 各挖空一部分
    for col, frac in (("age", 0.08), ("income", 0.12), ("score", 0.05)):
        idx = rng.choice(df.index, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2) 重复行
    dup_idx = rng.choice(df.index, size=10, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 3) 类型混入：income 里塞几个字符串
    df["income"] = df["income"].astype(object)
    str_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[str_idx, "income"] = ["未知", "--", "N/A", "", "待定"]

    # 4) 异常值：income 与 age 各塞极端值
    out_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[out_idx[:2], "income"] = [-99999, 999999]
    df.loc[out_idx[2:], "age"] = [-1, 200]

    # 5) 类别列污染：前后空格 / 大小写 / 别名
    df["city"] = df["city"].astype(object)
    pollute_idx = rng.choice(df.index, size=12, replace=False)
    aliases = [" 北京", "北京市", "shanghai", "SH ", "广州市", "深圳 ", "杭州 ", "beijing"]
    for i, idx in enumerate(pollute_idx):
        df.loc[idx, "city"] = aliases[i % len(aliases)]

    return df


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = make_sample()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[OK] 已生成示例数据: {OUT}")
    print(f"     形状: {df.shape[0]} 行 x {df.shape[1]} 列")


if __name__ == "__main__":
    main()
