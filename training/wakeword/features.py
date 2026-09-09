"""把增强后的音频批量转成 openWakeWord 的嵌入特征（每 80 ms 一帧 96 维），写成 memmap .npy。

产物（<workdir>/features/）：positive_train.npy / positive_val.npy / negative_train.npy / negative_val.npy，
形状都是 (N, 16, 96) float32——2 秒窗口正好 16 帧，与 openWakeWord 自带模型的输入一致。
"""

from __future__ import annotations

import hashlib
import logging
import os
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .augment import SR, AudioPool, Augmenter, load_audio_16k, load_rirs, to_int16, trim_to_speech
from .config import TrainingConfig
from .tts import ClipRecord, read_manifest, split_for

log = logging.getLogger(__name__)


def audio_features(ncpu: int = 1):
    from openwakeword.utils import AudioFeatures

    return AudioFeatures(inference_framework="onnx", ncpu=ncpu)


def embedding_shape(clip_seconds: float) -> tuple[int, int]:
    F = audio_features()
    return tuple(F.get_embedding_shape(clip_seconds))  # type: ignore[return-value]


def compute_features(
    batches: Iterable[np.ndarray], n_total: int, out_path: Path, clip_seconds: float, ncpu: int = 1
) -> Path:
    """batches 里每个元素是 (B, samples) 的 int16；写入 out_path，行数不足时最后截断。"""
    from tqdm import tqdm

    F = audio_features(ncpu=ncpu)
    n_frames, n_feat = F.get_embedding_shape(clip_seconds)
    tmp = out_path.with_suffix(".tmp.npy")
    fp = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_total, n_frames, n_feat))
    row = 0
    for batch in tqdm(batches, total=None, desc=f"features → {out_path.name}", unit="batch"):
        if row >= n_total:
            break
        emb = F.embed_clips(batch, batch_size=len(batch), ncpu=ncpu)
        emb = emb[: n_total - row]
        fp[row : row + len(emb)] = emb
        row += len(emb)
    fp.flush()
    del fp
    if row < n_total:
        src = np.load(tmp, mmap_mode="r")
        out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(row, n_frames, n_feat))
        out[:] = src[:row]
        out.flush()
        del out, src
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)
    log.info("wrote %s: %d rows", out_path, row)
    return out_path


# ---------------------------------------------------------------- 样本来源
class Sample:
    __slots__ = ("path", "positive", "weight", "kind", "text")

    def __init__(self, path: Path, positive: bool, weight: int = 1, kind: str = "", text: str = ""):
        self.path, self.positive, self.weight = path, positive, weight
        self.kind, self.text = (
            kind,
            text,
        )  # 来源类别（positive/adversarial/general/extra）与文本，仅用于评估报告


def extra_dir_samples(
    dirs: list[str], positive: bool, val_fraction: float, weight: int
) -> dict[str, list[Sample]]:
    """真人录音目录：按文件名哈希分 train/val；weight 是增强轮数的倍率（真录音更宝贵，多转几轮）。"""
    out: dict[str, list[Sample]] = {"train": [], "val": []}
    for d in dirs:
        for p in sorted(Path(d).rglob("*.wav")):
            key = hashlib.sha1(p.name.encode()).hexdigest()
            out[split_for(key, val_fraction)].append(Sample(p, positive, weight, kind="extra", text=p.stem))
    return out


def gather_samples(cfg: TrainingConfig, records: list[ClipRecord]) -> dict[tuple[str, str], list[Sample]]:
    """返回 {(label, split): [Sample]}，label ∈ {positive, negative}。"""
    groups: dict[tuple[str, str], list[Sample]] = {
        (lb, sp): [] for lb in ("positive", "negative") for sp in ("train", "val")
    }
    for r in records:
        lb = "positive" if r.is_positive else "negative"
        groups[(lb, r.split)].append(Sample(cfg.clips_dir / r.wav, r.is_positive, kind=r.label, text=r.text))
    for split, samples in extra_dir_samples(
        cfg.data.extra_positive_dirs, True, cfg.data.val_fraction, 3
    ).items():
        groups[("positive", split)] += samples
    for split, samples in extra_dir_samples(
        cfg.data.extra_negative_dirs, False, cfg.data.val_fraction, 2
    ).items():
        groups[("negative", split)] += samples
    return groups


def iter_augmented(
    samples: list[Sample], rounds: int, augmenter: Augmenter, batch_size: int, seed: int
) -> Iterator[np.ndarray]:
    """每轮把所有样本打乱后逐条增强，按 batch 产出 int16 (B, samples)。"""
    rng = np.random.default_rng(seed)
    cache: dict[Path, np.ndarray] = {}
    expanded = [s for s in samples for _ in range(s.weight)]
    for _ in range(rounds):
        order = rng.permutation(len(expanded))
        batch: list[np.ndarray] = []
        for idx in order:
            s = expanded[idx]
            if s.path not in cache:
                cache[s.path] = load_sample_audio(s)
            batch.append(augmenter(cache[s.path], s.positive))
            if len(batch) == batch_size:
                yield to_int16(np.stack(batch))
                batch = []
        if batch:
            yield to_int16(np.stack(batch))


def load_sample_audio(s: Sample) -> np.ndarray:
    """真人录音（kind == "extra"）先裁到有声段；TTS 片段合成时已经裁过。"""
    audio = load_audio_16k(s.path)
    return trim_to_speech(audio) if s.kind == "extra" else audio


def n_examples(samples: list[Sample], rounds: int) -> int:
    return rounds * sum(s.weight for s in samples)


# ---------------------------------------------------------------- 入口
def build_augmenter(
    cfg: TrainingConfig, records: list[ClipRecord], rir_dir: Path | None, seed: int
) -> Augmenter:
    background = (
        AudioPool.from_dirs(cfg.augment.background_dirs, cache=cfg.features_dir / "background_pool.npy")
        if cfg.augment.background_dirs
        else None
    )
    if background is None:
        log.warning(
            "no background_dirs configured: only synthetic noise / babble / RIR will be used. "
            "Recording some real room noise with the target device helps a lot."
        )
    babble_clips = [
        load_audio_16k(cfg.clips_dir / r.wav) for r in records if not r.is_positive and r.split == "train"
    ][:400]
    babble = AudioPool.from_clips(babble_clips)
    rirs = load_rirs(rir_dir)
    return Augmenter(cfg.augment, background=background, babble=babble, rirs=rirs, seed=seed)


def run_features(
    cfg: TrainingConfig, rir_dir: Path | None, ncpu: int = 0, overwrite: bool = False
) -> dict[str, Path]:
    records = read_manifest(cfg.manifest_path)
    if not records:
        raise SystemExit(f"no clips in {cfg.manifest_path}; run `synth` first")
    ncpu = ncpu or max(1, (os.cpu_count() or 2) // 2)
    cfg.features_dir.mkdir(parents=True, exist_ok=True)
    groups = gather_samples(cfg, records)
    outputs: dict[str, Path] = {}
    for (label, split), samples in groups.items():
        out = cfg.features_dir / f"{label}_{split}.npy"
        outputs[f"{label}_{split}"] = out
        if out.exists() and not overwrite:
            log.info("%s exists, skipping (use --overwrite to regenerate)", out)
            continue
        if not samples:
            log.warning("no samples for %s/%s", label, split)
            continue
        rounds = cfg.augment.rounds_positive if label == "positive" else cfg.augment.rounds_negative
        seed = cfg.train.seed + zlib.crc32(f"{label}/{split}".encode()) % 1000
        augmenter = build_augmenter(cfg, records, rir_dir, seed)
        batches = iter_augmented(samples, rounds, augmenter, cfg.augment.batch_size, seed)
        compute_features(batches, n_examples(samples, rounds), out, cfg.augment.clip_seconds, ncpu)
    return outputs


__all__ = ["SR", "compute_features", "embedding_shape", "run_features", "build_augmenter", "gather_samples"]
