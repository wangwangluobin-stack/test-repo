"""报告与图表生成。

输出:
- outputs/report.md
- outputs/figures/missing.png
- outputs/figures/hist.png
- outputs/figures/corr.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能出图

import matplotlib.pyplot as plt
import pandas as pd

from loader import overview

# 中文字体候选（Windows 常见）
CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = False
for _f in CJK_FONTS:
    plt.rcParams["font.sans-serif"] = [_f]
    break


def plot_missing(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """缺失值分布条形图（针对原始数据，传入清洗前的数据）。"""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    missing.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Missing values by column")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    path = out_dir / "missing.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_hist(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """数值列分布直方图。"""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None

    n = min(len(num_cols), 6)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, num_cols[:n]):
        ax.hist(df[c].dropna(), bins=20, color="#55A868", edgecolor="white")
        ax.set_title(c, fontsize=10)

    fig.tight_layout()
    path = out_dir / "hist.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_corr(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """数值列相关性热力图。"""
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return None

    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 标注数值
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")

    ax.set_title("Correlation matrix")
    fig.tight_layout()
    path = out_dir / "corr.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def build_report(raw: pd.DataFrame, cleaned: pd.DataFrame, log: dict, fig_dir: Path) -> str:
    """生成 Markdown 报告文本。"""
    ov_raw = overview(raw)
    ov_cln = overview(cleaned)

    lines: list[str] = []
    lines.append("# 数据清洗与探索性分析报告\n")

    # --- 1. 数据概览 ---
    lines.append("## 1. 数据概览\n")
    lines.append(f"- 原始数据：**{ov_raw['rows']} 行 × {ov_raw['cols']} 列**")
    lines.append(f"- 清洗后：**{ov_cln['rows']} 行 × {ov_cln['cols']} 列**")
    lines.append(f"- 内存占用：原始 {ov_raw['memory_kb']} KB → 清洗后 {ov_cln['memory_kb']} KB\n")

    lines.append("| 字段 | 原始类型 | 清洗后类型 |")
    lines.append("|---|---|---|")
    for c in ov_raw["columns"]:
        d_before = ov_raw["dtypes"].get(c, "-")
        d_after = ov_cln["dtypes"].get(c, "-")
        lines.append(f"| {c} | {d_before} | {d_after} |")
    lines.append("")

    # --- 2. 清洗动作 ---
    lines.append("## 2. 清洗过程\n")
    if "removed_duplicates" in log:
        lines.append(f"- 删除重复行：**{log['removed_duplicates']}** 行")
    if log.get("categorical_normalized"):
        for c, n in log["categorical_normalized"].items():
            lines.append(f"- 类别列 `{c}` 归一化（去空格/统一写法）：修改 **{n}** 处")
    if log.get("coerced_to_numeric"):
        for c, n in log["coerced_to_numeric"].items():
            lines.append(f"- 列 `{c}` 强制转数值，其中 **{n}** 个非数值被置为缺失")
    if log.get("outliers_nan"):
        for c, n in log["outliers_nan"].items():
            lines.append(f"- 列 `{c}` IQR 检测出异常值 **{n}** 个，已置为缺失")
    if log.get("missing_before"):
        detail = ", ".join(f"{k}={v}" for k, v in log["missing_before"].items())
        lines.append(f"- 缺失值填充前：{detail}")
    if log.get("rows_dropped"):
        lines.append(f"- 因缺失值删除 **{log['rows_dropped']}** 行")
    lines.append("")

    # --- 3. 描述统计 ---
    lines.append("## 3. 数值字段描述统计\n")
    desc = cleaned.describe().round(2)
    lines.append("| 统计量 | " + " | ".join(str(c) for c in desc.columns) + " |")
    lines.append("|---" * (len(desc.columns) + 1) + "|")
    for idx in desc.index:
        vals = " | ".join(f"{desc.loc[idx, c]}" for c in desc.columns)
        lines.append(f"| {idx} | {vals} |")
    lines.append("")

    # --- 4. 图表 ---
    lines.append("## 4. 可视化\n")
    for name, title in (("missing.png", "缺失值分布"),
                        ("hist.png", "数值分布"),
                        ("corr.png", "相关性矩阵")):
        p = fig_dir / name
        if p.exists():
            lines.append(f"### {title}\n")
            lines.append(f"![{title}](figures/{name})\n")

    # --- 5. 结论 ---
    lines.append("## 5. 结论与建议\n")
    lines.append(f"- 清洗后数据完整度为 **{100 * (1 - cleaned.isna().sum().sum() / max(cleaned.size, 1)):.2f}%**")
    total_out = sum(log.get("outliers_nan", {}).values())
    lines.append(f"- 共处理异常值 {total_out} 个、重复行 {log.get('removed_duplicates', 0)} 行")
    lines.append("- 后续可基于清洗后的数据做建模或分组统计（按 `city` / `gender` 分组对比）")
    lines.append("")

    return "\n".join(lines)


def save_report(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
