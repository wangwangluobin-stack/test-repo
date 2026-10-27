# data-cleaner

一个用于** CSV 数据清洗与自动分析报告**的小工具。

输入一份 CSV，输出：清洗后的数据 + 数据质量报告 + 可视化图表。

## 解决什么问题

拿到一份脏数据（缺失值、重复行、异常值、类型混乱）时，通常要手写一堆 pandas 代码。
这个工具把常见清洗流程固化下来，一条命令跑完，并自动生成一份可以交给老师 / 写进简历的分析报告。

## 功能

- 读取 CSV（自动识别编码与分隔符）
- 数据概览：行列数、字段类型、内存占用
- 缺失值统计与处理（删除 / 均值 / 中位数 / 众数 可配置）
- 重复行检测与去重
- 数值列异常值检测（IQR 方法）
- 自动生成图表：缺失值分布、数值列分布直方图、相关性热力图
- 输出 Markdown 分析报告

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 生成一份示例脏数据
python src/make_sample.py

# 4. 运行清洗 + 分析
python src/main.py --input data/raw/sample.csv
```

运行结束后在 `outputs/` 下会得到：

```
outputs/
├── clean.csv        # 清洗后的数据
├── report.md        # 分析报告
└── figures/
    ├── missing.png
    ├── hist.png
    └── corr.png
```

## 项目结构

```
data-cleaner/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── main.py          # 命令行入口
│   ├── loader.py        # 读取与编码探测
│   ├── clean.py         # 清洗逻辑
│   ├── report.py        # 报告与图表生成
│   └── make_sample.py   # 生成示例脏数据
├── tests/
│   └── test_clean.py    # 单元测试
├── data/
│   └── raw/             # 原始数据（不入库）
└── outputs/             # 输出结果（不入库）
```

## 依赖

- Python 3.9+
- pandas
- numpy
- matplotlib

## 说明

本项目为个人学习项目，用于练习 Python 数据分析与工程化（模块拆分、测试、Git 工作流）。
