# Codex 任务清单

这个文件是给 ChatGPT Codex（云环境）用的。按顺序执行，每个任务都要求**开 PR、不要直接推 main**。

## 使用方式

在 Codex 里贴下面每一条，一次一个。每条都让它开 PR，你 review diff 后再合并。

---

## 任务 1：补齐项目工程化配置

> 请阅读这个仓库的 README.md 和 src/ 下的代码，做以下事，开一个 PR：
> 1. 添加 `pyproject.toml`，配置项目元数据与 pytest 配置
> 2. 添加 GitHub Actions workflow（`.github/workflows/ci.yml`），在 push 和 PR 时自动运行 `pytest`
> 3. 添加 `CONTRIBUTING.md`，写明本地开发流程（虚拟环境、安装依赖、跑测试）
> 4. 检查 `.gitignore` 是否漏了什么，补充完整
>
> 每个文件加完后，用一句话说明为什么需要它。

---

## 任务 2：增强异常值处理

> 当前 `src/clean.py` 的 `remove_outliers` 只会把异常值置为 NaN。请开一个 PR 增强它：
> 1. 支持三种策略：`nan`（置空，现有行为）、`clip`（截断到边界值）、`drop`（删行）
> 2. 通过 CLI 参数 `--outlier-strategy` 暴露，默认保持 `nan`
> 3. 为三种策略各写单元测试
> 4. 在 README 的功能列表里更新说明
>
> 请在 PR 描述里说明每种策略适用的场景（比如 clip 适合什么数据、drop 会有什么代价）。

---

## 任务 3：增加分组统计

> 请开一个 PR，给报告增加"分组统计"章节：
> 1. 在 `src/report.py` 新增函数，按类别列（如 city、gender）对数值列做 groupby 聚合（count / mean / median）
> 2. 支持的类别列自动从数据中识别（非数值、且唯一值数量 < 20 的列）
> 3. 结果以 Markdown 表格写进 report.md 的第 3.5 节
> 4. 加一个柱状图对比不同分组的均值，输出到 figures/groupby.png
> 5. 写单元测试

---

## 任务 4：增加数据质量评分

> 请开一个 PR，实现一个"数据质量评分"功能：
> 1. 在 `src/clean.py`（或新建 `src/quality.py`）实现 `quality_score(df) -> dict`
> 2. 评分维度：完整度（缺失值占比）、唯一性（重复行占比）、一致性（类型是否统一）、有效性（异常值占比）
> 3. 输出 0-100 的总分 + 各维度得分与权重说明
> 4. 在报告开头展示质量评分，并给出"清洗前 vs 清洗后"对比
> 5. 写单元测试
>
> 权重怎么定请在 PR 里说明理由，不要随便拍。

---

## 任务 5：支持 Excel 与多编码

> 请开一个 PR，让 `src/loader.py` 同时支持 `.xlsx` / `.xls`：
> 1. `load_csv` 改名为 `load_data`（保留 `load_csv` 作为别名，不破坏现有调用）
> 2. 根据文件后缀自动选择 `pd.read_excel` 或 `pd.read_csv`
> 3. 新增依赖 `openpyxl`，更新 requirements.txt
> 4. 补充测试（用 tmp_path 生成一个临时 xlsx 测试）
> 5. 更新 README

---

## 学习要求（给自己看的）

每合并一个 PR，你要能回答：

- 这个 PR 改了哪些文件？为什么这么分模块？
- 新增的测试覆盖了什么边界情况？
- 如果不用 Codex，你自己会怎么写？哪一步你写不出来？

答不上来的，回去问 Codex，让它逐行解释，别只点合并。
