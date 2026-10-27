"""让独立运行的测试也能导入仓库现有 src 脚本模块。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
