from __future__ import annotations

import sys
from pathlib import Path

# 让 tests 里能直接 import training.*（它不随包安装）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
