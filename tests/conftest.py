from __future__ import annotations

import sys
from pathlib import Path

# 让 tests 里能直接 import training.*（它不随包安装）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    """标了 network 的测试默认跳过；CATMAN_IO_NETWORK_TESTS=1 才跑（要能上网、可能要 SSL_CERT_FILE）。"""
    import os

    import pytest

    if os.environ.get("CATMAN_IO_NETWORK_TESTS"):
        return
    skip = pytest.mark.skip(reason="network test; set CATMAN_IO_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
