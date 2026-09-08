"""评估导出的 ONNX 模型：

1. 逐条流式跑验证集音频（用 openwakeword.Model.predict_clip，和线上完全一样的特征流水线），
   分别在"干净"和"增强后"的片段上算各阈值下的召回率 / 负样本误接受率；
2. 在 11 小时通用音频特征上算每小时误唤醒次数（上升沿计数）。
结果写到 <export_dir>/eval.json，同时打印一张表，方便选阈值。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .augment import load_audio_16k, to_int16
from .config import TrainingConfig
from .features import build_augmenter, gather_samples
from .model import onnx_predictor
from .tts import read_manifest

log = logging.getLogger(__name__)

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
FRAME_SECONDS = 0.08


def clip_scores(model_path: Path, clips: list[np.ndarray]) -> np.ndarray:
    """每条片段用流式方式跑一遍，取最高分。"""
    from openwakeword import Model

    oww = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    name = next(iter(oww.models))
    scores = []
    for x in clips:
        oww.reset()
        frames = oww.predict_clip(x, padding=1)
        scores.append(max(f[name] for f in frames) if frames else 0.0)
    return np.array(scores)


def fp_per_hour(
    model_path: Path, validation: Path, n_frames: int = 16, chunk: int = 16384
) -> dict[str, float]:
    predict = onnx_predictor(model_path, threads=2)
    fp_val = np.load(validation, mmap_mode="r")
    T = len(fp_val)
    hours = T * FRAME_SECONDS / 3600
    out: dict[str, float] = {"hours": round(hours, 2)}
    for thr in THRESHOLDS:
        activations = 0
        prev = False
        for start in range(0, T - n_frames + 1, chunk):
            block = np.asarray(fp_val[start : min(T, start + chunk + n_frames - 1)], dtype=np.float32)
            win = np.transpose(np.lib.stride_tricks.sliding_window_view(block, n_frames, axis=0), (0, 2, 1))
            p = predict(np.ascontiguousarray(win)) >= thr
            edges = np.concatenate([[prev], p])
            activations += int(((~edges[:-1]) & edges[1:]).sum())
            prev = bool(p[-1])
        out[f"activations_per_hour@{thr}"] = round(activations / hours, 3)
    return out


def run_evaluation(
    cfg: TrainingConfig,
    model_path: Path | None,
    validation: Path | None,
    rir_dir: Path | None,
    max_clips: int = 400,
) -> dict:
    model_path = model_path or cfg.export_dir / f"{cfg.model_name}.onnx"
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    records = read_manifest(cfg.manifest_path)
    groups = gather_samples(cfg, records)
    rng = np.random.default_rng(cfg.train.seed + 7)
    result: dict = {"model": str(model_path)}

    def pick(samples, n):
        idx = rng.permutation(len(samples))[:n]
        return [samples[i] for i in idx]

    pos = pick(groups[("positive", "val")], max_clips)
    neg = pick(groups[("negative", "val")], max_clips)
    if not pos or not neg:
        raise SystemExit("validation split is empty; check val_fraction / manifest")

    pos_audio = [load_audio_16k(s.path) for s in pos]
    neg_audio = [load_audio_16k(s.path) for s in neg]
    augmenter = build_augmenter(cfg, records, rir_dir, seed=cfg.train.seed + 99)

    sets = {
        "clean": (pos_audio, neg_audio),
        "augmented": ([augmenter(x, True) for x in pos_audio], [augmenter(x, False) for x in neg_audio]),
    }
    for name, (p_audio, n_audio) in sets.items():
        ps = clip_scores(model_path, [to_int16(x) for x in p_audio])
        ns = clip_scores(model_path, [to_int16(x) for x in n_audio])
        kinds = sorted({s.kind for s in neg})
        result[name] = {
            "n_positive": len(ps),
            "n_negative": len(ns),
            **{f"recall@{t}": round(float((ps >= t).mean()), 4) for t in THRESHOLDS},
            **{f"false_accept@{t}": round(float((ns >= t).mean()), 4) for t in THRESHOLDS},
            "false_accept_by_kind@0.5": {
                k: round(float(np.mean([sc >= 0.5 for s, sc in zip(neg, ns, strict=True) if s.kind == k])), 4)
                for k in kinds
            },
            "positive_score_p10": round(float(np.percentile(ps, 10)), 4),
            "negative_score_p99": round(float(np.percentile(ns, 99)), 4),
        }
        if name == "clean":
            accepted = sorted(
                ((float(sc), s.text) for s, sc in zip(neg, ns, strict=True) if sc >= 0.5), reverse=True
            )
            result["top_false_accepts"] = [{"score": round(sc, 3), "text": t} for sc, t in accepted[:30]]
    if validation is not None and Path(validation).exists():
        result["fp_validation"] = fp_per_hour(model_path, Path(validation))

    out = model_path.with_name("eval.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():  # 评估结果并入模型元数据，install 时一起带走
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["evaluation"] = {k: v for k, v in result.items() if k != "model"}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(result))
    log.info("wrote %s", out)
    return result


def format_report(r: dict) -> str:
    lines = [f"model: {r['model']}", "", f"{'set':<10}{'thr':>6}{'recall':>9}{'false-accept':>14}"]
    for name in ("clean", "augmented"):
        if name not in r:
            continue
        for t in THRESHOLDS:
            lines.append(
                f"{name:<10}{t:>6}{r[name][f'recall@{t}']:>9.3f}{r[name][f'false_accept@{t}']:>14.3f}"
            )
    if "clean" in r:
        by_kind = r["clean"]["false_accept_by_kind@0.5"]
        lines.append("")
        lines.append(
            "false-accept@0.5 by negative kind: " + ", ".join(f"{k} {v:.3f}" for k, v in by_kind.items())
        )
    if r.get("top_false_accepts"):
        top = "; ".join(f"{d['text']} ({d['score']:.2f})" for d in r["top_false_accepts"][:8])
        lines.append("most accepted negatives: " + top)
    if "fp_validation" in r:
        fv = r["fp_validation"]
        lines.append("")
        lines.append(f"false activations per hour on {fv['hours']} h of generic audio:")
        lines.append(
            "  " + "  ".join(f"thr {t}: {fv[f'activations_per_hour@{t}']:.2f}/h" for t in THRESHOLDS)
        )
    return "\n".join(lines)
