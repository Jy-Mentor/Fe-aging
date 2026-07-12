"""pytest 配置"""

import sys
from pathlib import Path

# 确保 src/ 在 sys.path 中
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))