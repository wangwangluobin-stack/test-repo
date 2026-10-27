"""命令行入口。

用法:
    python src/main.py --input data/raw/sample.csv
    python src/main.py --input data/raw/sample.csv --fill mean
    python src/main.py --input data/raw/sample.csv --no-outliers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 python src/main.py 运行时找到同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import load_csv  # noqa: E402
from clean import run_pipeline, FILL_STRATEGIES  # noqa: E402
from report import build_report, save_report, plot_missing, plot_hist, plot_corr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV 数据清洗与自动分析报告")
    p.add_argument("--input", required=True, help="输入 CSV 路径")
    p.add_argument("--fill", default="median", choices=FILL_STRATEGIES,
                   help="缺失值填充策略（默认 median）")
    p.add_argument("--no-outliers", action="store_true",
                   help="跳过异常值处理")
    p.add_argument("--outdir", default=str(OUT_DIR), help="输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 读取数据: {args.input}")
    raw = load_csv(args.input)
    print(f"      {raw.shape[0]} 行 x {raw.shape[1]} 列")

    print(f"[2/4] 清洗中 (fill={args.fill}, outliers={not args.no_outliers})")
    cleaned, log = run_pipeline(
        raw,
        fill_strategy=args.fill,
        handle_outliers=not args.no_outliers,
    )
    print(f"      清洗后: {cleaned.shape[0]} 行 x {cleaned.shape[1]} 列")

    print("[3/4] 生成图表")
    plot_missing(raw, fig_dir)
    plot_hist(cleaned, fig_dir)
    plot_corr(cleaned, fig_dir)

    print("[4/4] 写出结果")
    clean_path = out_dir / "clean.csv"
    cleaned.to_csv(clean_path, index=False, encoding="utf-8-sig")

    report_text = build_report(raw, cleaned, log, fig_dir)
    report_path = out_dir / "report.md"
    save_report(report_text, report_path)

    print("\n完成:")
    print(f"  - {clean_path}")
    print(f"  - {report_path}")
    print(f"  - {fig_dir}")


if __name__ == "__main__":
    main()
