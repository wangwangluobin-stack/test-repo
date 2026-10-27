"""交互式 AI 数据分析助手（规则驱动，不调用模型 API）。"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from analyzer import analyze
from clean import FILL_STRATEGIES
from main import OUT_DIR, run_analysis
from qa import HELP_TEXT, apply_filter, answer_question


class AssistantSession:
    """持有数据、上轮列名、叠加筛选和最近 50 轮成功问答。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)
        self.context: dict = {}
        self.history: list[dict] = []

    def ask(self, question: str) -> dict:
        result = answer_question(self.df, question, self.context)
        self.context = result["context"]
        self.history.append({"question": question, "answer": result["answer"]})
        self.history = self.history[-50:]
        return result

    def discoveries(self) -> dict:
        current = self.df
        for spec in self.context.get("filters", []):
            current = apply_filter(current, spec)
        return analyze(current)


def _display(result: dict, output) -> None:
    print(result["answer"], file=output)
    if isinstance(result["data"], (list, dict)):
        print(json.dumps(result["data"], ensure_ascii=False, indent=2, allow_nan=False), file=output)


def run_chat(session: AssistantSession, input_stream=None, output_stream=None) -> None:
    input_stream = sys.stdin if input_stream is None else input_stream
    output = sys.stdout if output_stream is None else output_stream
    print("AI 数据分析助手（本地规则问答）\n" + HELP_TEXT, file=output)
    print("命令：发现 / history / help / 重置筛选 / exit；Ctrl-D 或 Ctrl-C 退出。", file=output)
    while True:
        try:
            print("\n你> ", end="", file=output, flush=True)
            line = input_stream.readline()
            if not line:
                break
            question = line.strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q", "退出"}:
                break
            if question.lower() in {"history", "历史"}:
                for item in session.history:
                    print(f"你：{item['question']}\n助手：{item['answer']}", file=output)
                continue
            if question.lower() in {"发现", "分析", "analyze"}:
                analysis = session.discoveries()
                print("当前筛选范围的自动发现：", file=output)
                for finding in analysis["findings"]:
                    print("- " + finding["message"], file=output)
                if not analysis["findings"]:
                    print("没有足够的数据生成自动发现。", file=output)
                continue
            _display(session.ask(question), output)
        except ValueError as error:
            print(f"无法回答：{error}", file=output)
        except KeyboardInterrupt:
            break
    print("\n会话结束。", file=output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI 数据分析助手：自动探索、质量评分和多轮规则问答")
    parser.add_argument("--input", required=True, help="输入 CSV 文件")
    parser.add_argument("--outdir", default=str(OUT_DIR), help="报告与清洗结果目录")
    parser.add_argument("--fill", default="median", choices=FILL_STRATEGIES)
    parser.add_argument("--no-outliers", action="store_true", help="跳过异常值处理")
    parser.add_argument("--raw", action="store_true", help="问答使用清洗前数据；报告仍比较清洗前后")
    parser.add_argument("--question", action="append", help="直接提问，可多次使用；省略则进入交互模式")
    args = parser.parse_args(argv)
    try:
        raw, cleaned, _ = run_analysis(args.input, args.outdir, args.fill, not args.no_outliers)
        session = AssistantSession(raw if args.raw else cleaned)
        print("\n问答数据：" + ("清洗前原始数据" if args.raw else "清洗后数据"))
        if args.question:
            for question in args.question:
                _display(session.ask(question), sys.stdout)
        else:
            run_chat(session)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
