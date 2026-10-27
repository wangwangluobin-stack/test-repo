from io import StringIO

import pandas as pd
import pytest

from agent import AssistantSession, main, run_chat


def test_session_remembers_columns_and_filters_without_changing_source():
    df = pd.DataFrame({"age": [20, 30, 40], "income": [100, 200, 300]})
    session = AssistantSession(df)
    df.loc[0, "income"] = 999
    assert session.ask("平均收入")["data"] == 200
    assert session.ask("那中位数呢")["data"] == 200
    session.ask("年龄大于20")
    assert session.ask("平均收入")["data"] == 250
    assert session.discoveries()["overview"]["rows"] == 2
    saved = session.context.copy()
    with pytest.raises(ValueError):
        session.ask("不支持的问题")
    assert session.context == saved
    assert len(session.history) == 4


def test_chat_handles_errors_history_discoveries_exit_and_eof():
    session = AssistantSession(pd.DataFrame({"age": [20, 30, 40]}))
    commands = StringIO("\n平均年龄\n那中位数呢\n不支持的问题\nhistory\n发现\nhelp\nexit\n")
    output = StringIO()
    run_chat(session, commands, output)
    text = output.getvalue()
    assert "mean = 30.0" in text
    assert "median = 30.0" in text
    assert "无法回答" in text
    assert "你：平均年龄" in text
    assert "当前筛选范围的自动发现" in text
    assert "会话结束" in text
    run_chat(session, StringIO(""), StringIO())


def test_session_history_is_bounded():
    session = AssistantSession(pd.DataFrame({"age": [20]}))
    for _ in range(55):
        session.ask("有多少行")
    assert len(session.history) == 50


def test_cli_generates_full_report_and_answers_multiple_questions(tmp_path, capsys):
    source = tmp_path / "input.csv"
    pd.DataFrame({"age": [20, 30, 40], "income": [100, 200, 300],
                  "city": ["北京", "上海", "北京"]}).to_csv(source, index=False)
    output = tmp_path / "outputs"
    code = main(["--input", str(source), "--outdir", str(output),
                 "--question", "平均收入", "--question", "那中位数呢"])
    assert code == 0
    text = (output / "report.md").read_text(encoding="utf-8")
    assert "清洗前 vs 清洗后" in text
    assert "## 自动发现" in text
    assert "### 类别列频次" in text
    assert "### 强相关特征对" in text
    assert (output / "clean.csv").is_file()
    assert (output / "figures" / "corr.png").read_bytes().startswith(b'\x89PNG')
    console = capsys.readouterr().out
    assert "mean = 200.0" in console
    assert "median = 200.0" in console


def test_cli_missing_file_has_useful_error(tmp_path, capsys):
    assert main(["--input", str(tmp_path / "missing.csv"), "--outdir", str(tmp_path / "out")]) == 1
    assert "文件不存在" in capsys.readouterr().err


def test_cli_raw_mode_and_pipeline_options_are_forwarded(monkeypatch, capsys):
    raw = pd.DataFrame({"age": [20, 20, 30]})
    cleaned = pd.DataFrame({"age": [20, 30]})
    calls = []

    def prepare(*args):
        calls.append(args)
        return raw, cleaned, {}

    monkeypatch.setattr("agent.run_analysis", prepare)
    assert main(["--input", "input.csv", "--outdir", "out", "--fill", "mean",
                 "--no-outliers", "--raw", "--question", "有多少行"]) == 0
    assert calls == [("input.csv", "out", "mean", False)]
    assert "共有 3 行" in capsys.readouterr().out


def test_chat_keyboard_interrupt_exits_cleanly():
    class Interrupt:
        def readline(self):
            raise KeyboardInterrupt

    output = StringIO()
    run_chat(AssistantSession(pd.DataFrame()), Interrupt(), output)
    assert "会话结束" in output.getvalue()


def test_repeated_pipeline_run_removes_stale_figures(tmp_path):
    from main import run_analysis

    source = tmp_path / "input.csv"
    output = tmp_path / "out"
    pd.DataFrame({"x": [1, 2, None, 4], "y": [2, 4, 6, 8]}).to_csv(source, index=False)
    run_analysis(source, output)
    assert {p.name for p in (output / "figures").glob("*.png")} == {"missing.png", "hist.png", "corr.png"}
    pd.DataFrame({"city": ["北京", "上海"]}).to_csv(source, index=False)
    run_analysis(source, output)
    assert list((output / "figures").glob("*.png")) == []
    assert "![" not in (output / "report.md").read_text(encoding="utf-8")
