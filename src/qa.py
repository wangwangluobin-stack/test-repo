"""规则问答：将受支持的问法映射到固定 pandas 操作，不执行用户代码。"""

from __future__ import annotations

import json
import math
import re

import pandas as pd

from analyzer import analyze, numeric_columns


ALIASES = {
    "年龄": ("age",), "收入": ("income",), "得分": ("score",), "分数": ("score",),
    "城市": ("city",), "性别": ("gender",), "日期": ("signup_date", "date"),
    "用户编号": ("user_id", "id"),
}

HELP_TEXT = (
    "支持：数据概览、行数、列数、缺失值、重复行、均值、中位数、总和、最小值、最大值、"
    "标准差、不同值数量、类别频次、相关性、前 N 行、条件筛选、分组统计。\n"
    "示例：平均年龄；收入的中位数；城市频次；年龄和收入的相关性；收入最高的前5行；"
    "年龄大于30；按城市分组计算收入平均值。\n"
    "追问：那中位数呢？（沿用上一列）；筛选后提问沿用筛选结果，输入“重置筛选”恢复全部数据。\n"
    "列名可直接使用 CSV 中的名称；也支持示例数据的年龄/收入/城市等中文别名。"
)

AGGREGATIONS = (
    ("mean", r"平均|均值|\bmean\b|\baverage\b"),
    ("median", r"中位数|\bmedian\b"),
    ("sum", r"总和|合计|求和|\bsum\b"),
    ("std", r"标准差|\bstd\b"),
    ("min", r"最小|最低|\bmin\b|\bminimum\b"),
    ("max", r"最大|最高|\bmax\b|\bmaximum\b"),
)

OPERATORS = {
    ">=": ">=", "≤": "<=", "≥": ">=", "<=": "<=", "!=": "!=", "==": "==",
    ">": ">", "<": "<", "大于等于": ">=", "小于等于": "<=", "不等于": "!=",
    "大于": ">", "小于": "<", "等于": "==", "超过": ">", "高于": ">",
    "低于": "<", "至少": ">=", "至多": "<=",
}
OP_PATTERN = re.compile("|".join(re.escape(op) for op in sorted(OPERATORS, key=len, reverse=True)))


def _column_mentions(df: pd.DataFrame, question: str) -> list[str]:
    matches = []
    occupied: list[tuple[int, int]] = []
    terms = [(column, column) for column in df.columns]
    for alias, candidates in ALIASES.items():
        present = [c for c in candidates if c in df.columns]
        # 真实列名优先；含糊别名不猜测。
        if len(present) == 1 and alias not in df.columns:
            terms.append((alias, present[0]))
    for term, column in sorted(terms, key=lambda item: len(item[0]), reverse=True):
        pattern = re.escape(term)
        if term.isascii():
            pattern = r"(?<![a-zA-Z0-9_])" + pattern + r"(?![a-zA-Z0-9_])"
        for match in re.finditer(pattern, question, re.IGNORECASE):
            start, end = match.span()
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            matches.append((start, column))
    return list(dict.fromkeys(column for _, column in sorted(matches)))


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


def _number(value: object) -> float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def apply_filter(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """应用单个受限条件；会话发现与问答共用同一筛选规则。"""
    column, operator, value = spec["column"], spec["operator"], spec["value"]
    if column not in df.columns or operator not in {">", "<", ">=", "<=", "==", "!="}:
        raise ValueError("筛选上下文无效，请重置筛选")
    series = df[column]
    if isinstance(value, (int, float)):
        if column not in numeric_columns(df):
            raise ValueError(f"{column} 不是数值列，无法进行数值条件筛选")
    elif operator not in {"==", "!="}:
        raise ValueError("文本列仅支持等于或不等于")
    mask = {">": series.gt, "<": series.lt, ">=": series.ge,
            "<=": series.le, "==": series.eq, "!=": series.ne}[operator](value)
    return df.loc[mask.fillna(False) & series.notna()]


def _parse_filter(df: pd.DataFrame, question: str) -> tuple[dict, str] | None:
    quoted_spans = [match.span() for match in re.finditer(r'''"[^"]*"|'[^']*'|“[^”]*”''', question)]
    matches = [match for match in OP_PATTERN.finditer(question)
               if not any(start <= match.start() < end for start, end in quoted_spans)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("每次支持一个筛选条件；可连续提问叠加筛选")
    match = matches[0]
    columns = _column_mentions(df, question[:match.start()])
    if len(columns) != 1:
        raise ValueError("筛选时请在比较符前明确一个列名，如：年龄大于30")
    tail = question[match.end():].strip()
    value_match = re.match(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", tail)
    if value_match:
        value = float(value_match.group())
        if not math.isfinite(value):
            raise ValueError("筛选值必须是有限数值")
        remaining = tail[value_match.end():]
    else:
        quoted = re.match(r'''["'“](.*?)["'”]''', tail)
        plain = re.match(r".+?(?=的|有多少|有几|共有|[\s，,。？?]|$)", tail)
        if quoted:
            value, remaining = quoted.group(1), tail[quoted.end():]
        elif plain:
            value, remaining = plain.group(), tail[plain.end():]
        else:
            raise ValueError("筛选缺少比较值，如：城市等于\"北京\"")
    return {"column": columns[0], "operator": OPERATORS[match.group()], "value": value}, remaining


def answer_question(df: pd.DataFrame, question: str, context: dict | None = None) -> dict:
    """返回 intent、answer、data 与新 context；仅显式追问复用上一列，筛选持续生效。

    支持 17 类常见问法。无法识别/歧义/列类型不匹配时抛 ValueError，不猜测问题含义。
    context 是本函数返回的会话元数据，不包含可执行代码；输入数据与 context 均不修改。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("问答需要 pandas.DataFrame")
    if not df.columns.is_unique or not all(isinstance(c, str) and c for c in df.columns):
        raise ValueError("问答要求列名为非空且唯一的字符串")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入问题；输入 help 查看支持的问法")
    question = question.strip()
    lowered = question.lower()
    context = context or {}
    state = {"columns": list(context.get("columns", [])),
             "filters": [dict(spec) for spec in context.get("filters", [])]}

    def respond(intent: str, data: object, text: str, columns: list[str] | None = None) -> dict:
        if columns is not None:
            state["columns"] = columns
        state["intent"] = intent
        return {"intent": intent, "answer": text, "data": data, "context": state}

    if lowered in {"help", "帮助", "?", "？"}:
        return respond("help", None, HELP_TEXT)
    if re.fullmatch(r"重置筛选|重置|全部数据|reset|clear filters", lowered):
        state = {"columns": [], "filters": []}
        return respond("reset", {"rows": len(df)}, f"已恢复全部 {len(df)} 行数据。")

    frame = df
    for spec in state["filters"]:
        frame = apply_filter(frame, spec)
    parsed = _parse_filter(df, question)
    if parsed:
        spec, remaining = parsed
        frame = apply_filter(frame, spec)
        state["filters"].append(spec)
        state["columns"] = [spec["column"]]
        # 一次筛选后，可直接问均值/行数，也可在下一轮提问。
        if remaining.strip(" 的，,。？?呢"):
            followup = remaining.strip(" 的，,。？?呢")
            if not _column_mentions(df, followup):
                followup = f"{spec['column']} {followup}"
            return answer_question(df, followup, state)
        return respond("filter", {"rows": len(frame), "preview": _records(frame.head(10))},
                       f"筛选后共有 {len(frame)} 行（预览前 10 行）；后续问题将沿用此筛选。")

    columns = _column_mentions(df, question)
    if not columns and re.search(r"它|该列|这列|那|呢|\bit\b|\bthat\b|what about", lowered):
        columns = [c for c in state["columns"] if c in df.columns]

    if re.search(r"概览|概况|有哪些列|列名|\boverview\b|\bdescribe\b", lowered):
        data = {"rows": len(frame), "columns": list(frame.columns),
                "dtypes": {c: str(frame[c].dtype) for c in frame.columns}}
        return respond("overview", data, f"当前数据 {len(frame)} 行、{len(frame.columns)} 列。")
    if re.search(r"多少列|几列|列数|column count", lowered):
        return respond("column_count", len(frame.columns), f"共有 {len(frame.columns)} 列。")
    if re.search(r"缺失|空值|\bmissing\b|\bnull\b", lowered):
        counts = frame[columns or list(frame.columns)].isna().sum()
        data = {str(c): int(n) for c, n in counts.items()}
        return respond("missing", data, f"缺失单元格共 {sum(data.values())} 个。", columns or None)
    if re.search(r"重复|\bduplicates?\b", lowered):
        count = int(frame.duplicated().sum()) if len(frame.columns) else 0
        return respond("duplicates", count, f"完全重复的多余行有 {count} 行。")

    aggregation = next((name for name, pattern in AGGREGATIONS if re.search(pattern, lowered)), None)
    if aggregation is None and columns and re.search(r"那|呢|what about", lowered):
        last_intent = context.get("intent")
        if last_intent in {name for name, _ in AGGREGATIONS}:
            aggregation = last_intent
    if re.search(r"分组|\bgroup\s+by\b|^各", lowered):
        if not columns:
            raise ValueError("请指定分组列，如：按城市分组计算收入平均值")
        group_column = columns[0]
        grouped = frame.groupby(group_column, dropna=False, observed=True, sort=False)
        if re.search(r"数量|多少|计数|\bcount\b", lowered) and not aggregation:
            count_name = "count" if group_column != "count" else "row_count"
            data = _records(grouped.size().rename(count_name).reset_index().head(100))
        else:
            if len(columns) != 2 or aggregation is None:
                raise ValueError("分组统计需指定分组列、数值列和统计方法")
            value_column = columns[1]
            if value_column not in numeric_columns(frame):
                raise ValueError(f"{value_column} 不是数值列")
            finite = frame.copy()
            finite[value_column] = finite[value_column].replace([float("inf"), -float("inf")], float("nan"))
            grouped_values = finite.groupby(group_column, dropna=False, observed=True, sort=False)[value_column]
            values = grouped_values.sum(min_count=1) if aggregation == "sum" else getattr(grouped_values, aggregation)()
            name = f"{value_column}_{aggregation}"
            if name == group_column:
                name = "result_" + name
            data = _records(values.rename(name).reset_index().head(100))
        return respond("groupby", data, "分组统计结果（最多显示前 100 组）。", columns)

    if re.search(r"相关|\bcorrel", lowered):
        if not columns:
            pairs = analyze(frame)["correlations"]
            return respond("correlation", pairs, f"发现 {len(pairs)} 对强相关特征（|r|≥0.7）；相关不代表因果。")
        if len(columns) != 2 or any(c not in numeric_columns(frame) for c in columns):
            raise ValueError("相关性问题需指定两个数值列")
        pair = frame[columns].replace([float("inf"), -float("inf")], float("nan")).dropna()
        if len(pair) < 3 or any(pair[c].nunique() < 2 for c in columns):
            raise ValueError("相关性至少需要 3 对有限观测，且两个列均非常量")
        r = _number(pair[columns[0]].astype(float).corr(pair[columns[1]].astype(float)))
        if r is None:
            raise ValueError("当前数值尺度无法稳定计算相关性")
        return respond("correlation", {"coefficient": r, "observations": len(pair)},
                       f"{columns[0]} 与 {columns[1]} 的 Pearson 相关系数为 {r:.4f}；相关不代表因果。", columns)

    if re.search(r"多少行|几行|行数|多少条|多少人|多少记录|row count|\brows\b", lowered) and not re.search(r"前\s*\d|\btop\s*\d", lowered):
        return respond("row_count", len(frame), f"共有 {len(frame)} 行。")

    top = re.search(r"前\s*(\d+)|\btop\s*(\d+)", lowered)
    if len(columns) != 1:
        raise ValueError("请明确指定一个列名；相关性和分组问题可以指定两个列")
    column = columns[0]
    if top:
        n = int(top.group(1) or top.group(2))
        if not 1 <= n <= 100:
            raise ValueError("前 N 行的 N 必须在 1–100 之间")
        ascending = bool(re.search(r"最低|最小|升序|\blowest\b|\bsmallest\b", lowered))
        data = _records(frame.sort_values(column, ascending=ascending, na_position="last").head(n))
        return respond("top", data, f"按 {column} {'升序' if ascending else '降序'}排列的前 {n} 行。", columns)
    if re.search(r"频次|频数|分布|最多|\bfrequency\b|value counts", lowered):
        counts = frame[column].value_counts(dropna=False).head(20)
        data = [{"value": None if pd.isna(value) else str(value), "count": int(n)} for value, n in counts.items()]
        return respond("frequency", data, f"{column} 的频次统计（最多显示前 20 项，包含缺失值）。", columns)
    if re.search(r"不同值|唯一值|种类|几种|\bunique\b|\bnunique\b", lowered):
        n = int(frame[column].nunique())
        return respond("unique", n, f"{column} 有 {n} 个不同的非空值。", columns)
    if aggregation:
        if column not in numeric_columns(frame):
            raise ValueError(f"{column} 不是数值列，请选择数值列进行统计")
        values = frame[column].replace([float("inf"), -float("inf")], float("nan")).dropna()
        value = _number(getattr(values, aggregation)()) if len(values) else None
        return respond(aggregation, value,
                       f"{column} 的 {aggregation} = {value}" if value is not None else f"{column} 无足够的有限数值，无法计算 {aggregation}。", columns)
    raise ValueError("暂不支持此问法；输入 help 查看支持的问法")
