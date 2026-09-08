"""下载训练用的公共资源（都来自 openWakeWord 作者在 Hugging Face 上的数据集）：

- MIT 房间冲激响应（270 条 16 kHz wav，约 8 MB）——模拟不同房间的混响
- validation_set_features.npy（约 180 MB）——11 小时的通用音频特征，用来估计每小时误唤醒次数
- openwakeword_features_ACAV100M_2000_hrs_16bit.npy 的**前 N 小时**——通用负样本特征。
  整个文件 17 GB，这里用 HTTP Range 只取开头一段（每小时约 15 MB），支持断点续传。
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger(__name__)

HF = "https://huggingface.co/datasets"
RIR_REPO = f"{HF}/davidscripka/MIT_environmental_impulse_responses"
FEATURES_REPO = f"{HF}/davidscripka/openwakeword_features"
VALIDATION_URL = f"{FEATURES_REPO}/resolve/main/validation_set_features.npy"
ACAV_URL = f"{FEATURES_REPO}/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

# ACAV 特征文件里每一行是 16 帧 × 80 ms = 1.28 s
SECONDS_PER_ACAV_ROW = 16 * 0.08
CHUNK = 1 << 20


def parse_npy_header(head: bytes) -> tuple[int, dict]:
    """解析 .npy 文件头，返回 (数据起始偏移, header 字典)。只支持 1.0/2.0 版本。"""
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = head[6]
    if major == 1:
        hlen = int.from_bytes(head[8:10], "little")
        start = 10
    else:
        hlen = int.from_bytes(head[8:12], "little")
        start = 12
    header = ast.literal_eval(head[start : start + hlen].decode("latin1"))
    return start + hlen, header


def _stream_to(url: str, dest: Path, start: int = 0, end: int | None = None, desc: str = "") -> None:
    """把 url 的 [start, end] 字节流式写到 dest（追加模式，用于断点续传）。"""
    from tqdm import tqdm

    headers = {}
    if start or end is not None:
        headers["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if headers and r.status_code != 206:
            raise RuntimeError(f"server ignored Range request for {url} (status {r.status_code})")
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dest, "ab") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    offset = tmp.stat().st_size if tmp.exists() else 0
    _stream_to(url, tmp, start=offset, desc=desc or dest.name)
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------- RIR
def download_rirs(resources_dir: Path) -> Path:
    out = resources_dir / "mit_rirs"
    if out.exists() and any(out.glob("*.wav")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    listing = requests.get(
        f"{RIR_REPO.replace('/datasets/', '/api/datasets/')}/tree/main/16khz", timeout=60
    ).json()
    files = [item["path"] for item in listing if item["path"].endswith(".wav")]
    log.info("downloading %d MIT room impulse responses", len(files))
    from tqdm import tqdm

    for path in tqdm(files, desc="RIR"):
        dest = out / Path(path).name
        if dest.exists():
            continue
        r = requests.get(f"{RIR_REPO}/resolve/main/{path}", timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return out


# ---------------------------------------------------------------- 验证集特征
def download_validation_features(resources_dir: Path) -> Path:
    resources_dir.mkdir(parents=True, exist_ok=True)
    return download_file(VALIDATION_URL, resources_dir / "validation_set_features.npy", "validation features")


# ---------------------------------------------------------------- ACAV 负样本切片
def download_precomputed_negatives(resources_dir: Path, hours: float) -> Path | None:
    """只下载 ACAV100M 特征文件的前 `hours` 小时，写成一个独立的合法 .npy（float16, (N, 16, 96)）。"""
    if hours <= 0:
        return None
    resources_dir.mkdir(parents=True, exist_ok=True)
    n_rows = int(hours * 3600 / SECONDS_PER_ACAV_ROW)
    dest = resources_dir / f"acav_negatives_{n_rows}rows.npy"
    if dest.exists():
        return dest

    head = requests.get(ACAV_URL, headers={"Range": "bytes=0-1023"}, timeout=60)
    head.raise_for_status()
    data_offset, header = parse_npy_header(head.content)
    shape, descr = header["shape"], header["descr"]
    if len(shape) != 3 or header.get("fortran_order"):
        raise RuntimeError(f"unexpected ACAV layout: {header}")
    n_rows = min(n_rows, shape[0])
    row_bytes = int(np.dtype(descr).itemsize) * shape[1] * shape[2]
    log.info(
        "fetching %d ACAV rows (%.1f h, %.0f MB) of %d",
        n_rows,
        n_rows * SECONDS_PER_ACAV_ROW / 3600,
        n_rows * row_bytes / 1e6,
        shape[0],
    )

    raw = resources_dir / (dest.name + ".raw")
    offset = raw.stat().st_size if raw.exists() else 0
    need = n_rows * row_bytes
    if offset < need:
        _stream_to(
            ACAV_URL, raw, start=data_offset + offset, end=data_offset + need - 1, desc="ACAV negatives"
        )
    if raw.stat().st_size != need:
        raise RuntimeError(f"short download: {raw.stat().st_size} != {need} bytes; delete {raw} and retry")

    out = np.lib.format.open_memmap(
        dest, mode="w+", dtype=np.dtype(descr), shape=(n_rows, shape[1], shape[2])
    )
    src = np.memmap(raw, dtype=np.dtype(descr), mode="r", shape=(n_rows, shape[1], shape[2]))
    step = 4096
    for i in range(0, n_rows, step):
        out[i : i + step] = src[i : i + step]
    out.flush()
    del out, src
    raw.unlink()
    (dest.with_suffix(".json")).write_text(
        json.dumps({"source": ACAV_URL, "rows": n_rows, "hours": n_rows * SECONDS_PER_ACAV_ROW / 3600})
    )
    return dest


# ---------------------------------------------------------------- 入口
def prepare_resources(cfg) -> dict[str, Path | None]:
    """按配置下载所需资源，返回各资源路径。"""
    rdir = cfg.resources_dir
    rdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path | None] = {"rir_dir": None, "validation": None, "precomputed": None}
    if cfg.augment.rir_dirs:
        out["rir_dir"] = Path(cfg.augment.rir_dirs[0])
    elif cfg.augment.download_rirs:
        out["rir_dir"] = download_rirs(rdir)
    if cfg.data.fp_validation:
        out["validation"] = download_validation_features(rdir)
    if cfg.data.precomputed_negative_hours > 0:
        out["precomputed"] = download_precomputed_negatives(rdir, cfg.data.precomputed_negative_hours)
    return out


def copy_tree_wavs(src: Path, dst: Path) -> int:
    """把目录里的 wav 平铺复制到 dst（供 extra_*_dirs 用）。"""
    n = 0
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*.wav"):
        shutil.copy2(p, dst / p.name)
        n += 1
    return n
