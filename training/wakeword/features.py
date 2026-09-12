"""把增强后的音频批量转成 openWakeWord 的嵌入特征（每 80 ms 一帧 96 维），写成 memmap .npy。

产物（<workdir>/features/）：positive_train.npy / positive_val.npy / negative_train.npy / negative_val.npy
是合成片段；真人录音（`data.extra_*_dirs`）单独写成 extra_positive_train.npy 等（重新划分、改权重时不用
重算几千条合成片段的特征），train 时拼在一起。形状都是 (N, 16, 96) float32——2 秒窗口正好 16 帧，
与 openWakeWord 自带模型的输入一致。
"""

from __future__ import annotations

import logging
import os
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .augment import SR, AudioPool, Augmenter, load_audio_16k, load_rirs, to_int16, trim_to_speech
from .config import TrainingConfig
from .tts import ClipRecord, read_manifest

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
    __slots__ = ("path", "positive", "weight", "kind", "text", "rate", "voice")

    def __init__(
        self,
        path: Path,
        positive: bool,
        weight: int = 1,
        kind: str = "",
        text: str = "",
        rate: str = "",
        voice: str = "",
    ):
        self.path, self.positive, self.weight = path, positive, weight
        # 来源类别（positive/adversarial/homophone/general/extra）、文本、TTS 语速与音色，仅用于评估报告
        self.kind, self.text, self.rate, self.voice = kind, text, rate, voice


def extra_dir_samples(dirs: list[str], positive: bool, weight: int) -> list[Sample]:
    """真人录音目录：递归收所有 wav，子目录名当说话人（voice）；weight 是增强轮数的倍率。

    目录不存在直接报错——配置写错了却悄悄训出一个没有真人录音的模型，比报错糟得多。
    """
    out: list[Sample] = []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            raise SystemExit(f"recording directory not found: {d}")
        for p in sorted(root.rglob("*.wav")):
            voice = p.parent.name if p.parent != root else root.name
            out.append(Sample(p, positive, weight, kind="extra", text=p.stem, voice=voice))
    return out


def gather_samples(cfg: TrainingConfig, records: list[ClipRecord]) -> dict[str, list[Sample]]:
    """返回 {特征文件名: [Sample]}。

    positive_train / positive_val / negative_train / negative_val 是合成片段；
    extra_positive_train / extra_positive_val / extra_negative_train / extra_negative_val 是真人录音
    （`data.extra_*_dirs` 全部训练，`data.extra_*_val_dirs` 只验证），没配置时为空列表。
    """
    groups: dict[str, list[Sample]] = {
        f"{lb}_{sp}": [] for lb in ("positive", "negative") for sp in ("train", "val")
    }
    holdout = set(cfg.data.holdout_voices)
    for r in records:
        lb = "positive" if r.is_positive else "negative"
        split = r.split
        if (r.label == "homophone" and not cfg.data.train_on_homophones) or r.voice in holdout:
            split = "val"  # 近音短语（默认）与留出的音色只评估、不训练
        groups[f"{lb}_{split}"].append(
            Sample(
                cfg.clips_dir / r.wav, r.is_positive, kind=r.label, text=r.text, rate=r.rate, voice=r.voice
            )
        )
    d = cfg.data
    groups["extra_positive_train"] = extra_dir_samples(d.extra_positive_dirs, True, d.extra_positive_weight)
    groups["extra_positive_val"] = extra_dir_samples(d.extra_positive_val_dirs, True, 1)
    groups["extra_negative_train"] = extra_dir_samples(d.extra_negative_dirs, False, d.extra_negative_weight)
    groups["extra_negative_val"] = extra_dir_samples(d.extra_negative_val_dirs, False, 1)
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
    for name, samples in groups.items():
        out = cfg.features_dir / f"{name}.npy"
        if not samples:
            if name.startswith("extra_"):
                if out.exists():  # 配置里去掉了真人录音目录：旧特征不能留着被 train 悄悄拼进去
                    out.unlink()
                    log.info("removed stale %s", out)
            else:
                log.warning("no samples for %s", name)
            continue
        outputs[name] = out
        if out.exists() and not overwrite:
            log.info("%s exists, skipping (use --overwrite to regenerate)", out)
            continue
        rounds = cfg.augment.rounds_positive if "positive" in name else cfg.augment.rounds_negative
        seed = cfg.train.seed + zlib.crc32(name.encode()) % 1000
        augmenter = build_augmenter(cfg, records, rir_dir, seed)
        batches = iter_augmented(samples, rounds, augmenter, cfg.augment.batch_size, seed)
        compute_features(batches, n_examples(samples, rounds), out, cfg.augment.clip_seconds, ncpu)
    return outputs


__all__ = ["SR", "compute_features", "embedding_shape", "run_features", "build_augmenter", "gather_samples"]
