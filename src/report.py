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
from matplotlib import font_manager
import pandas as pd

from loader import overview
from quality import LABELS, quality_score
from analyzer import analyze, numeric_columns


def _cell(value: object) -> str:
    """避免列名、类别中的 Markdown 分隔符破坏新报告表格。"""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _format_stat(value: float | None) -> str:
    return "无数据" if value is None else f"{value:.3f}"


# 中文字体候选（Windows 常见）
CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = False
_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
_cjk_fonts = [font for font in CJK_FONTS if font in _available_fonts]
if _cjk_fonts:
    plt.rcParams["font.sans-serif"] = _cjk_fonts


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
    num_cols = numeric_columns(df)
    if not num_cols:
        return None

    n = min(len(num_cols), 6)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, num_cols[:n]):
        finite = df[c].replace([float("inf"), -float("inf")], float("nan")).dropna().astype(float)
        ax.hist(finite, bins=20, color="#55A868", edgecolor="white")
        ax.set_title(c, fontsize=10)

    fig.tight_layout()
    path = out_dir / "hist.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_corr(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """数值列相关性热力图。"""
    num = df[numeric_columns(df)].replace([float("inf"), -float("inf")], float("nan")).astype(float)
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


def _quality_section(before: dict, after: dict) -> list[str]:
    """渲染评分对比，计算逻辑留在 quality 层。"""
    lines: list[str] = []
    change = after["total_score"] - before["total_score"]
    lines.append("## 数据质量评分：清洗前 vs 清洗后\n")
    lines.append(
        f"**总分：清洗前 {before['total_score']:.2f}/100 → "
        f"清洗后 {after['total_score']:.2f}/100（变化 {change:+.2f} 分）**\n"
    )
    lines.append("| 维度 | 权重 | 清洗前 | 清洗后 | 变化 |")
    lines.append("|---|---|---|---|---|")
    for key, weight in before["weights"].items():
        old, new = before["scores"][key], after["scores"][key]
        lines.append(f"| {LABELS[key]} | {weight:.0%} | {old:.2f} | {new:.2f} | {new - old:+.2f} |")
    lines.append("")
    lines.append("权重依据（启发式默认值，未经业务数据校准）：\n")
    for key, reason in before["weight_reasons"].items():
        lines.append(f"- {LABELS[key]} {reason}")
    lines.append("")
    lines.append(
        "评分口径：完整度为非 NA 单元格占比；唯一性为去重后行数占比；"
        "一致性为各列主导语义类型的非 NA 单元格占比（数值字符串与实数归为同类，布尔值单独分类）；"
        "有效性为可解析数值中有限且未被每列 1.5 × IQR 标记的占比。"
        "少于 4 个有限数值的列只检查有限性，无数值时有效性记 100，表示未发现数值问题。\n"
    )
    lines.append(
        "缺失值只按 pandas NA 判断，不识别空白字符串或文本占位符。"
        "无可评估数据（空表或全为 NA）时所有维度和总分记 0。"
        "评分不验证业务真值、主键或日期文本格式，填充后分数上升也不代表恢复了真实数据。\n"
    )
    if not before["has_data"] or not after["has_data"]:
        states = [name for name, result in (("清洗前", before), ("清洗后", after)) if not result["has_data"]]
        lines.append(f"无可评估数据：{'、'.join(states)}。\n")

    return lines


def _discovery_section(analysis: dict) -> list[str]:
    """渲染结构化探索结果，不在报告层重新推断结论。"""
    lines: list[str] = []
    lines.append("## 自动发现（基于清洗后数据）\n")
    if analysis["findings"]:
        for finding in analysis["findings"]:
            lines.append(f"- {_cell(finding['message'])}")
    else:
        lines.append("没有足够的数据生成自动发现。")
    lines.append("")

    lines.append("### 数值列分布\n")
    lines.append("| 字段 | 有限观测 | 均值 | 中位数 | 标准差 | 分布判断 | IQR 异常值 |")
    lines.append("|---|---|---|---|---|---|---|")
    for column, stats in analysis["numeric"].items():
        lines.append(
            f"| {_cell(column)} | {stats['finite_count']} | {_format_stat(stats['mean'])} | "
            f"{_format_stat(stats['median'])} | {_format_stat(stats['std'])} | "
            f"{stats['distribution']['description']} | {stats['outliers']} |"
        )
    if not analysis["numeric"]:
        lines.append("\n没有实数类型的数值列。")
    lines.append("")

    lines.append("### 类别列频次\n")
    for column, stats in analysis["categorical"].items():
        lines.append(f"#### {_cell(column)}（非空不同值 {stats['unique']} 个）\n")
        lines.append("| 值 | 次数 | 占全部行比例 |")
        lines.append("|---|---|---|")
        for item in stats["frequencies"]:
            label = "（缺失）" if item["value"] is None else _cell(item["value"])
            lines.append(f"| {label} | {item['count']} | {item['share']:.1%} |")
        if stats["truncated"]:
            lines.append("\n仅展示频次最高的 10 项。")
        lines.append("")
    if not analysis["categorical"]:
        lines.append("没有类别列。\n")

    lines.append("### 强相关特征对\n")
    lines.append("| 特征一 | 特征二 | Pearson r | 成对有效观测 |")
    lines.append("|---|---|---|---|")
    for pair in analysis["correlations"]:
        lines.append(f"| {_cell(pair['left'])} | {_cell(pair['right'])} | "
                     f"{pair['coefficient']:.3f} | {pair['observations']} |")
    if not analysis["correlations"]:
        lines.append("\n没有满足 |r| ≥ 0.7 且至少有 3 对有效观测的特征对。")
    lines.append("\n相关不代表因果；数值标识符可能产生没有业务意义的相关性。\n")

    return lines


def build_report(raw: pd.DataFrame, cleaned: pd.DataFrame, log: dict, fig_dir: Path) -> str:
    """生成 Markdown 报告文本。"""
    ov_raw = overview(raw)
    ov_cln = overview(cleaned)

    lines: list[str] = []
    lines.append("# 数据清洗与探索性分析报告\n")

    before = quality_score(raw)
    after = quality_score(cleaned)
    lines.extend(_quality_section(before, after))
    lines.extend(_discovery_section(analyze(cleaned)))

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
    if cleaned.shape[1]:
        desc = cleaned.describe().round(2)
        lines.append("| 统计量 | " + " | ".join(str(c) for c in desc.columns) + " |")
        lines.append("|---" * (len(desc.columns) + 1) + "|")
        for idx in desc.index:
            vals = " | ".join(f"{desc.loc[idx, c]}" for c in desc.columns)
            lines.append(f"| {idx} | {vals} |")
    else:
        lines.append("没有可统计的字段。")
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
    lines.append(f"- 清洗后数据完整度为 **{after['scores']['completeness']:.2f}%**")
    total_out = sum(log.get("outliers_nan", {}).values())
    lines.append(f"- 共处理异常值 {total_out} 个、重复行 {log.get('removed_duplicates', 0)} 行")
    lines.append("- 建议结合业务核查缺失、重复与统计异常，避免把相关关系解释为因果。")
    lines.append("")

    return "\n".join(lines)


def save_report(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
