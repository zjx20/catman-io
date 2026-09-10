"""可选的粤语 ASR 模型（sherpa-onnx 发布的 int8 ONNX）与下载。

模型不随包（一两百 MB），用 ``catman-io setup --asr [名字]`` 下载到 ``<data_dir>/models/asr/``。
"""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"


@dataclass(frozen=True)
class AsrModel:
    name: str
    kind: str  # sensevoice | wenet_ctc
    dirname: str
    size_mb: int
    note: str
    model_file: str = "model.int8.onnx"
    tokens_file: str = "tokens.txt"

    @property
    def url(self) -> str:
        return f"{RELEASE}{self.dirname}.tar.bz2"


ASR_MODELS: dict[str, AsrModel] = {
    m.name: m
    for m in (
        AsrModel(
            "sense-voice-2025-09",
            "sensevoice",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
            166,
            "SenseVoice-Small，用 2.18 万小时粤语数据微调过；不出标点",
        ),
        AsrModel(
            "sense-voice-2024-07",
            "sensevoice",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            163,
            "SenseVoice-Small 原版；use_itn 可出标点与数字",
        ),
        AsrModel(
            "wenet-yue-2025-09",
            "wenet_ctc",
            "sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10",
            117,
            "WenetSpeech-Yue 粤语专用 CTC 模型，最小最快",
        ),
    )
}

# backend 名 → 默认模型
DEFAULT_MODEL = {"sensevoice": "sense-voice-2025-09", "wenet_yue": "wenet-yue-2025-09"}


def resolve_model(backend: str, name: str = "") -> AsrModel:
    key = name or DEFAULT_MODEL.get(backend, "")
    if key not in ASR_MODELS:
        raise ValueError(f"unknown ASR model {key!r} (backend {backend!r}); known: {sorted(ASR_MODELS)}")
    return ASR_MODELS[key]


def model_path(root: Path, model: AsrModel) -> Path:
    return root / model.dirname


def is_installed(root: Path, model: AsrModel) -> bool:
    d = model_path(root, model)
    return (d / model.model_file).exists() and (d / model.tokens_file).exists()


def ensure_asr_model(root: Path, model: AsrModel, *, quiet: bool = False) -> Path:
    """没装就下载并解包；返回模型目录。"""
    d = model_path(root, model)
    if is_installed(root, model):
        return d
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{model.dirname}.tar.bz2"
    if not quiet:
        log.info("downloading %s (%d MB) from %s", model.name, model.size_mb, model.url)
    _download(model.url, archive, quiet=quiet)
    with tarfile.open(archive, "r:bz2") as tf:
        members = [m for m in tf.getmembers() if _safe_member(m, model.dirname)]
        tf.extractall(root, members=members)
    archive.unlink(missing_ok=True)
    if not is_installed(root, model):
        raise RuntimeError(f"archive did not contain {model.model_file}/{model.tokens_file}: {archive}")
    return d


def _safe_member(m: tarfile.TarInfo, prefix: str) -> bool:
    if not m.name.startswith(prefix + "/") or ".." in m.name.split("/"):
        return False
    return not (m.issym() or m.islnk())


def _download(url: str, dest: Path, *, quiet: bool) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "catman-io"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and not quiet:
                pct = done * 100 // total
                if pct // 10 != last // 10:
                    last = pct
                    log.info("  %d%%", pct)
    tmp.replace(dest)
