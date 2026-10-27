# 贡献指南

## 准备本地环境

需要 Python 3.9 或更新版本及 Git；CI 使用 Python 3.9 和 3.12。

```bash
git clone https://github.com/wangwangluobin-stack/test-repo.git
cd test-repo
git switch -c feat/your-change
python -m venv .venv
```

激活虚拟环境：

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Windows 命令提示符（cmd）可运行 `.venv\Scripts\activate.bat`。

在仓库根目录安装项目及测试依赖：

```bash
python -m pip install -e ".[dev]"
```

可编辑安装会让 `src/` 下的代码改动直接生效。`pyproject.toml` 声明项目元数据、
运行依赖和 `dev` 测试依赖；README 中的 `requirements.txt` 安装方式仍然可用。
调整依赖时，请同步这两个文件中的对应依赖约束。

## 运行与测试

以下命令均在仓库根目录执行：

```bash
# 生成示例数据，再运行完整清洗与报告流程
python src/make_sample.py
python src/main.py --input data/raw/sample.csv

# 运行全部测试，与 CI 保持一致
python -m pytest

# 开发时只运行一个文件或一个测试
python -m pytest tests/test_clean.py -v
python -m pytest tests/test_clean.py -k test_fill_missing_median
```

示例数据写入 `data/raw/sample.csv`；清洗结果、Markdown 报告和图表写入 `outputs/`，
这些生成文件不会被 Git 跟踪。不要提交真实用户数据或凭据。
pytest 的测试目录与模块搜索路径已在 `pyproject.toml` 配置，无需手动设置 `PYTHONPATH`。

## 提交改动

1. 在独立分支上开发，保持每次改动聚焦一个问题。
2. 修改功能时，在 `tests/` 中补充对应行为和边界情况的测试；必要时更新文档。
3. 运行 `python -m pytest`，并通过 `git diff --check` 检查空白问题。
4. 使用 `git status --short` 和 `git diff` 检查待提交内容，确认没有数据、凭据或生成文件。
5. 提交并推送分支，向 `main` 发起 PR；说明改动目的、实现和验证结果，等待 review 与 CI 通过后合并。

CI 会在 push 和 PR 时自动运行全部测试，请勿直接向 `main` 推送改动。
