"""scripts/wakeword_model.py：在临时仓库里把 publish → list → show → pull 走一遍。需要 git。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "wakeword_model.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def run(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and r.returncode != 0:
        raise AssertionError(f"{args} failed:\n{r.stdout}\n{r.stderr}")
    return r


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """一个有代码提交、配置文件、训练产物和合成片段的仓库，外加一个裸仓库当 origin。"""
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(remote)], check=True)
    repo = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    git(repo, "remote", "add", "origin", str(remote))
    (repo / "catman_io/wakeword/models").mkdir(parents=True)
    (repo / "catman_io/wakeword/models/VERSION").write_text("v1\n")
    (repo / "training/wakeword/configs").mkdir(parents=True)
    cfg = repo / "training/wakeword/configs/siu_maau_jan.yaml"
    cfg.write_text(
        "workdir: training/wakeword/work/tiny\n"
        "data:\n"
        "  extra_positive_dirs: [training/wakeword/real/tiny/positive]\n"
        "  extra_positive_val_dirs: [training/wakeword/real/tiny/positive_val]\n"
    )
    (repo / ".gitignore").write_text(
        "training/wakeword/work/\ntraining/wakeword/real/\ncatman_io/wakeword/models/*.onnx\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "code")
    git(repo, "push", "-q", "origin", "main")
    work = repo / "training/wakeword/work/tiny"
    (work / "export").mkdir(parents=True)
    (work / "clips").mkdir()
    (work / "resources").mkdir()
    (work / "features").mkdir()
    (work / "export/tiny.onnx").write_bytes(bytes(range(256)) * 4)
    meta = {
        "model_name": "tiny",
        "wake_phrase": "小貓人",
        "trained_at": "2026-01-01T00:00:00Z",
        "config": {
            "tts": {"rates": ["+0%"], "negative_rates": ["+0%"], "voices": ["a"]},
            "augment": {"p_speed": 0},
        },
        "evaluation": {"clean": {"recall@0.5": 0.9, "false_accept_by_kind@0.5": {"general": 0.0}}},
    }
    (work / "export/tiny.json").write_text(json.dumps(meta, ensure_ascii=False))
    (work / "clips/positive_x.wav").write_bytes(b"RIFF" + b"\0" * 64)
    (work / "clips/manifest.jsonl").write_text("{}\n")
    (work / "resources/big.npy").write_bytes(b"\0" * 32)
    (work / "features/positive_train.npy").write_bytes(b"\0" * 32)
    # 配置引用的真人录音目录（gitignore 了，但发布时要跟着版本走）
    for sub in ("positive/male", "positive_val/male"):
        (repo / "training/wakeword/real/tiny" / sub).mkdir(parents=True)
    (repo / "training/wakeword/real/tiny/positive/male/1.wav").write_bytes(b"RIFF" + b"\1" * 64)
    (repo / "training/wakeword/real/tiny/positive_val/male/2.wav").write_bytes(b"RIFF" + b"\2" * 64)
    return repo


def test_publish_pull_roundtrip(repo: Path, tmp_path: Path):
    out = run(repo, "publish", "v1", "--notes", "测试版本", "--push").stdout
    assert "created models/wakeword/v1" in out and "pushed" in out

    # 分支 = 代码提交 + 一个提交；带模型、配置、export、clips，不带 resources / features
    assert git(repo, "rev-list", "--count", "main..models/wakeword/v1").strip() == "1"
    files = set(git(repo, "ls-tree", "-r", "--name-only", "models/wakeword/v1").splitlines())
    assert "catman_io/wakeword/models/tiny.onnx" in files
    assert "catman_io/wakeword/models/tiny.json" in files
    assert "training/wakeword/work/tiny/clips/positive_x.wav" in files
    assert "training/wakeword/work/tiny/export/tiny.onnx" in files
    assert "MODEL.md" in files and "training/wakeword/configs/siu_maau_jan.yaml" in files
    assert "training/wakeword/real/tiny/positive/male/1.wav" in files
    assert "training/wakeword/real/tiny/positive_val/male/2.wav" in files
    assert not any("/resources/" in f or "/features/" in f for f in files)
    assert "training/wakeword/real/tiny" in git(repo, "log", "-1", "--format=%B", "models/wakeword/v1")
    # 工作区与当前分支没被动过
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert git(repo, "status", "--porcelain").strip() == ""

    # 同一版本不能重复发布，除非 --force
    assert run(repo, "publish", "v1", check=False).returncode != 0

    # 另一个克隆：list / show / pull（默认版本来自 VERSION 文件）
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo / ".." / "origin.git"), str(clone)], check=True)
    assert "v1" in run(clone, "list").stdout
    assert "小貓人" in run(clone, "show", "v1").stdout
    out = run(clone, "pull").stdout
    assert "installed wake-word model v1" in out
    dest = clone / "catman_io/wakeword/models"
    assert (dest / "tiny.onnx").read_bytes() == bytes(range(256)) * 4
    assert json.loads((dest / "tiny.json").read_text())["wake_phrase"] == "小貓人"
    assert (dest / "INSTALLED").read_text().startswith("v1 ")
    # pull 只取模型，不把训练数据拉进工作区
    assert not (clone / "training/wakeword/work").exists()

    # 指定目标目录也行
    run(clone, "pull", "v1", "--dest", str(tmp_path / "elsewhere"))
    assert (tmp_path / "elsewhere" / "tiny.onnx").exists()


def test_publish_without_data_and_bad_version(repo: Path):
    assert run(repo, "publish", "bad/name", check=False).returncode != 0
    run(repo, "publish", "v2", "--no-data")
    files = set(git(repo, "ls-tree", "-r", "--name-only", "models/wakeword/v2").splitlines())
    assert "catman_io/wakeword/models/tiny.onnx" in files
    assert "training/wakeword/work/tiny/export/tiny.json" in files
    assert not any("/clips/" in f or "/real/" in f for f in files)

    # 配置引用的录音目录不存在 / 不在仓库里：报错，不能悄悄发布一个缺数据的版本
    cfg = repo / "training/wakeword/configs/siu_maau_jan.yaml"
    cfg.write_text(
        "workdir: training/wakeword/work/tiny\ndata:\n  extra_positive_dirs: [training/wakeword/real/nope]\n"
    )
    r = run(repo, "publish", "v3", check=False)
    assert r.returncode != 0 and "does not exist" in r.stderr
    cfg.write_text(f"workdir: training/wakeword/work/tiny\ndata:\n  extra_positive_dirs: ['{repo.parent}']\n")
    r = run(repo, "publish", "v3", check=False)
    assert r.returncode != 0 and "outside the repository" in r.stderr
