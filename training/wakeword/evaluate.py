"""评估导出的 ONNX 模型：

1. 逐条流式跑验证集音频（用 openwakeword.Model.predict_clip，和线上完全一样的特征流水线），
   分别在"干净"和"增强后"的片段上算各阈值下的召回率 / 负样本误接受率；
2. 快语速压力测试：把干净片段用 WSOLA 变速不变调压到 1.25～2 倍速（音高不变，只是说得快），
   看召回掉多少——真人快说「小貓人」大约只有半秒，比 TTS 正常语速快一倍；
3. 真人录音验证集（`data.extra_*_val_dirs`）：整条录音不裁、不增强，流式打分，按说话人（子目录）报召回，
   并列出漏掉的那几条——这一栏最接近真机；
4. 在 11 小时通用音频特征上算每小时误唤醒次数（上升沿计数）。
结果写到 <export_dir>/eval.json，同时打印一张表，方便选阈值。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .augment import load_audio_16k, time_stretch, to_int16
from .config import TrainingConfig
from .features import build_augmenter, gather_samples, load_sample_audio
from .model import onnx_predictor
from .tts import read_manifest

log = logging.getLogger(__name__)

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
# 快语速压力测试的倍速（相对片段本身的语速；音高不变）
FAST_TEMPOS = (1.25, 1.5, 1.75, 2.0)
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

    # 合成片段抽 max_clips 条；真人录音验证集太宝贵，全部都要
    real_pos, real_neg = groups["extra_positive_val"], groups["extra_negative_val"]
    pos = real_pos + pick(groups["positive_val"], max_clips)
    neg = real_neg + pick(groups["negative_val"], max_clips)
    if not pos or not neg:
        raise SystemExit("validation split is empty; check val_fraction / manifest")

    pos_audio = [load_sample_audio(s) for s in pos]
    neg_audio = [load_sample_audio(s) for s in neg]
    augmenter = build_augmenter(cfg, records, rir_dir, seed=cfg.train.seed + 99)

    sets = {
        "clean": (pos_audio, neg_audio),
        "augmented": ([augmenter(x, True) for x in pos_audio], [augmenter(x, False) for x in neg_audio]),
    }
    for tempo in FAST_TEMPOS:
        sets[f"fast_x{tempo}"] = (
            [time_stretch(x, tempo) for x in pos_audio],
            [time_stretch(x, tempo) for x in neg_audio],
        )
    for name, (p_audio, n_audio) in sets.items():
        ps = clip_scores(model_path, [to_int16(x) for x in p_audio])
        ns = clip_scores(model_path, [to_int16(x) for x in n_audio])
        kinds = sorted({s.kind for s in neg})
        rates = sorted({s.rate for s in pos if s.rate}, key=_rate_key)
        voices = sorted({s.voice for s in pos if s.voice})
        hit = [(s, sc >= 0.5) for s, sc in zip(pos, ps, strict=True)]
        result[name] = {
            "n_positive": len(ps),
            "n_negative": len(ns),
            **{f"recall@{t}": round(float((ps >= t).mean()), 4) for t in THRESHOLDS},
            **{f"false_accept@{t}": round(float((ns >= t).mean()), 4) for t in THRESHOLDS},
            "false_accept_by_kind@0.5": {
                k: round(float(np.mean([sc >= 0.5 for s, sc in zip(neg, ns, strict=True) if s.kind == k])), 4)
                for k in kinds
            },
            "recall_by_rate@0.5": {
                r: round(float(np.mean([sc >= 0.5 for s, sc in zip(pos, ps, strict=True) if s.rate == r])), 4)
                for r in rates
            },
            "recall_by_voice@0.5": {
                v: round(float(np.mean([h for s, h in hit if s.voice == v])), 4) for v in voices
            },
            "positive_score_p10": round(float(np.percentile(ps, 10)), 4),
            "positive_score_median": round(float(np.median(ps)), 4),
            "negative_score_p99": round(float(np.percentile(ns, 99)), 4),
        }
        if name == "clean":
            accepted = sorted(
                ((float(sc), s.text) for s, sc in zip(neg, ns, strict=True) if sc >= 0.5), reverse=True
            )
            result["top_false_accepts"] = [{"score": round(sc, 3), "text": t} for sc, t in accepted[:30]]
    if real_pos or real_neg:
        result["real"] = real_recordings(model_path, real_pos, real_neg)
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


def real_recordings(model_path: Path, pos: list, neg: list) -> dict:
    """真人录音验证集：整条录音原样流式打分（不裁静音、不增强），和真机上听到的一样。"""
    ps = clip_scores(model_path, [to_int16(load_audio_16k(s.path)) for s in pos])
    ns = clip_scores(model_path, [to_int16(load_audio_16k(s.path)) for s in neg])
    voices = sorted({s.voice for s in pos})
    out: dict = {"n_positive": len(ps), "n_negative": len(ns)}
    if len(ps):
        out.update({f"recall@{t}": round(float((ps >= t).mean()), 4) for t in THRESHOLDS})
        out["recall_by_voice@0.5"] = {
            v: round(float(np.mean([sc >= 0.5 for s, sc in zip(pos, ps, strict=True) if s.voice == v])), 4)
            for v in voices
        }
        out["positive_score_median"] = round(float(np.median(ps)), 4)
        out["missed@0.5"] = [
            {"clip": f"{s.voice}/{s.path.name}", "score": round(float(sc), 3)}
            for s, sc in sorted(zip(pos, ps, strict=True), key=lambda x: x[1])
            if sc < 0.5
        ]
    if len(ns):
        out.update({f"false_accept@{t}": round(float((ns >= t).mean()), 4) for t in THRESHOLDS})
        out["negative_score_p99"] = round(float(np.percentile(ns, 99)), 4)
    return out


def _rate_key(rate: str) -> float:
    try:
        return float(rate.rstrip("%"))
    except ValueError:
        return 0.0


def format_report(r: dict) -> str:
    lines = [f"model: {r['model']}", "", f"{'set':<11}{'thr':>6}{'recall':>9}{'false-accept':>14}"]
    for name in ("clean", "augmented"):
        if name not in r:
            continue
        for t in THRESHOLDS:
            lines.append(
                f"{name:<11}{t:>6}{r[name][f'recall@{t}']:>9.3f}{r[name][f'false_accept@{t}']:>14.3f}"
            )
    fast = [f"fast_x{t}" for t in FAST_TEMPOS if f"fast_x{t}" in r]
    if fast:
        lines.append("")
        lines.append("fast-speech stress test (clean clips time-stretched, pitch kept):")
        lines.append(
            f"{'set':<11}{'recall@.3':>10}{'recall@.5':>10}{'recall@.7':>10}{'median':>8}{'fa@.5':>8}"
        )
        for name in fast:
            m = r[name]
            lines.append(
                f"{name:<11}{m['recall@0.3']:>10.3f}{m['recall@0.5']:>10.3f}{m['recall@0.7']:>10.3f}"
                f"{m['positive_score_median']:>8.3f}{m['false_accept@0.5']:>8.3f}"
            )
    if "clean" in r:
        by_kind = r["clean"]["false_accept_by_kind@0.5"]
        lines.append("")
        lines.append(
            "false-accept@0.5 by negative kind: " + ", ".join(f"{k} {v:.3f}" for k, v in by_kind.items())
        )
        by_rate = r["clean"].get("recall_by_rate@0.5", {})
        if by_rate:
            lines.append(
                "clean recall@0.5 by TTS rate: " + ", ".join(f"{k} {v:.3f}" for k, v in by_rate.items())
            )
        for name in ("clean", "fast_x1.5"):
            by_voice = r.get(name, {}).get("recall_by_voice@0.5", {})
            if by_voice:
                lines.append(
                    f"{name} recall@0.5 by voice: "
                    + ", ".join(f"{k.replace('Neural', '')} {v:.3f}" for k, v in by_voice.items())
                )
    if r.get("top_false_accepts"):
        top = "; ".join(f"{d['text']} ({d['score']:.2f})" for d in r["top_false_accepts"][:8])
        lines.append("most accepted negatives: " + top)
    real = r.get("real")
    if real:
        lines.append("")
        lines.append(
            f"real recordings (whole clips, streaming): {real['n_positive']} positive, "
            f"{real['n_negative']} negative"
        )
        if real["n_positive"]:
            lines.append(
                "  recall: "
                + "  ".join(f"@{t} {real[f'recall@{t}']:.3f}" for t in THRESHOLDS)
                + f"  (median score {real['positive_score_median']:.3f})"
            )
            lines.append(
                "  recall@0.5 by speaker: "
                + ", ".join(f"{k} {v:.3f}" for k, v in real["recall_by_voice@0.5"].items())
            )
            if real["missed@0.5"]:
                lines.append(
                    "  missed@0.5: "
                    + ", ".join(f"{d['clip']} ({d['score']:.2f})" for d in real["missed@0.5"])
                )
        if real["n_negative"]:
            lines.append(
                "  false-accept: " + "  ".join(f"@{t} {real[f'false_accept@{t}']:.3f}" for t in THRESHOLDS)
            )
    if "fp_validation" in r:
        fv = r["fp_validation"]
        lines.append("")
        lines.append(f"false activations per hour on {fv['hours']} h of generic audio:")
        lines.append(
            "  " + "  ".join(f"thr {t}: {fv[f'activations_per_hour@{t}']:.2f}/h" for t in THRESHOLDS)
        )
    return "\n".join(lines)
