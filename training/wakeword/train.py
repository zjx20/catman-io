"""训练小分类器。训练策略改写自 openWakeWord 的 `train.py`（Apache-2.0, David Scripka）：

1. 三段式训练：lr=1e-4 跑 `steps` 步，再各用 lr/10、lr/100 跑 steps/10 步；
2. 负样本权重在每一段里从 1 线性升到 max_negative_weight——先学会认唤醒词，再狠狠压误唤醒；
   某一段结束时验证集误唤醒仍高于目标，下一段把权重上限翻倍；
3. 每个 batch 只对"还没学会"的样本回传（负样本分数 ≥ 0.001、正样本分数 < 0.999）；
4. 用 11 小时通用音频的特征估计每小时误唤醒，定期存检查点，最后把最好的几个检查点权重平均。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import TrainingConfig
from .model import WakeWordNet, average_state_dicts, export_onnx

log = logging.getLogger(__name__)

FRAME_SECONDS = 0.08
THRESHOLD = 0.5


@dataclass
class Datasets:
    pos_train: np.ndarray
    neg_train: np.ndarray
    pos_val: np.ndarray
    neg_val: np.ndarray
    precomputed: np.ndarray | None = None  # (K, 16, 96) float16 memmap
    fp_val: np.ndarray | None = None  # (T, 96) float32 memmap
    n_frames: int = 16
    n_features: int = 96
    # 四个集合里各有多少行来自真人录音（增强后），只用于日志和元数据
    real: dict[str, int] = field(default_factory=dict)

    @property
    def fp_val_hours(self) -> float:
        return 0.0 if self.fp_val is None else len(self.fp_val) * FRAME_SECONDS / 3600


def load_datasets(cfg: TrainingConfig, precomputed: Path | None, validation: Path | None) -> Datasets:
    fdir = cfg.features_dir

    def load(name: str) -> np.ndarray:
        p = fdir / f"{name}.npy"
        if not p.exists():
            raise SystemExit(f"missing {p}; run `features` first")
        return np.load(p)

    def load_with_extra(name: str) -> tuple[np.ndarray, int]:
        """合成片段的特征 + 真人录音的特征（有就拼上）；返回拼好的数组和真人录音那部分的行数。"""
        base = load(name)
        extra = fdir / f"extra_{name}.npy"
        if not extra.exists():
            return base, 0
        x = np.load(extra)
        return np.concatenate([base, x]), len(x)

    pos_train, n_real_pos = load_with_extra("positive_train")
    neg_train, n_real_neg = load_with_extra("negative_train")
    pos_val, n_real_pos_val = load_with_extra("positive_val")
    neg_val, n_real_neg_val = load_with_extra("negative_val")
    ds = Datasets(pos_train, neg_train, pos_val, neg_val)
    ds.real = {
        "positive_train": n_real_pos,
        "negative_train": n_real_neg,
        "positive_val": n_real_pos_val,
        "negative_val": n_real_neg_val,
    }
    ds.n_frames, ds.n_features = ds.pos_train.shape[1], ds.pos_train.shape[2]
    if precomputed is not None and Path(precomputed).exists():
        ds.precomputed = np.load(precomputed, mmap_mode="r")
        if ds.precomputed.shape[1:] != (ds.n_frames, ds.n_features):
            raise SystemExit(
                f"precomputed negatives shape {ds.precomputed.shape} != (*, {ds.n_frames}, {ds.n_features})"
            )
    if validation is not None and Path(validation).exists():
        ds.fp_val = np.load(validation, mmap_mode="r")
    log.info(
        "datasets: pos %d/%d (real recordings %d/%d), neg %d/%d (real %d/%d), precomputed %s, fp-val %.1f h",
        len(ds.pos_train),
        len(ds.pos_val),
        ds.real["positive_train"],
        ds.real["positive_val"],
        len(ds.neg_train),
        len(ds.neg_val),
        ds.real["negative_train"],
        ds.real["negative_val"],
        0 if ds.precomputed is None else len(ds.precomputed),
        ds.fp_val_hours,
    )
    return ds


@dataclass
class Checkpoint:
    step: int
    stage: int
    state: dict[str, torch.Tensor]
    metrics: dict[str, float] = field(default_factory=dict)


def lr_warmup_cosine(step: int, total: int, warmup: int, hold: int, target_lr: float) -> float:
    if step < warmup:
        return target_lr * step / max(1, warmup)
    if step < warmup + hold:
        return target_lr
    progress = (step - warmup - hold) / max(1, total - warmup - hold)
    return 0.5 * target_lr * (1 + np.cos(np.pi * min(1.0, progress)))


class Trainer:
    def __init__(self, cfg: TrainingConfig, ds: Datasets):
        self.cfg, self.ds = cfg, ds
        t = cfg.train
        torch.manual_seed(t.seed)
        self.rng = np.random.default_rng(t.seed)
        threads = t.threads or max(1, os.cpu_count() or 1)
        torch.set_num_threads(threads)
        self.model = WakeWordNet(ds.n_frames, ds.n_features, t.layer_size, t.n_blocks, t.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=t.lr)
        self.checkpoints: list[Checkpoint] = []
        self.history: list[dict[str, float]] = []
        self._val_x, self._val_y = self._balanced_val()

    # ------------------------------------------------------------ 数据
    def _balanced_val(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.concatenate([self.ds.pos_val, self.ds.neg_val]).astype(np.float32)
        y = np.concatenate([np.ones(len(self.ds.pos_val)), np.zeros(len(self.ds.neg_val))]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        t, ds, rng = self.cfg.train, self.ds, self.rng
        parts, labels = [], []
        pi = rng.integers(0, len(ds.pos_train), t.batch_positive)
        parts.append(ds.pos_train[pi])
        labels.append(np.ones(len(pi)))
        ni = rng.integers(0, len(ds.neg_train), t.batch_negative)
        parts.append(ds.neg_train[ni])
        labels.append(np.zeros(len(ni)))
        if ds.precomputed is not None and t.batch_precomputed > 0:
            ki = np.sort(rng.integers(0, len(ds.precomputed), t.batch_precomputed))
            parts.append(np.asarray(ds.precomputed[ki], dtype=np.float32))
            labels.append(np.zeros(len(ki)))
        x = torch.from_numpy(np.concatenate(parts).astype(np.float32))
        y = torch.from_numpy(np.concatenate(labels).astype(np.float32))
        return x, y

    # ------------------------------------------------------------ 评估
    @torch.no_grad()
    def eval_balanced(self, model: WakeWordNet) -> dict[str, float]:
        model.eval()
        p = model(self._val_x).squeeze(1)
        y = self._val_y
        pred = p >= THRESHOLD
        pos = y == 1
        recall = float(pred[pos].float().mean()) if pos.any() else 0.0
        n_fp = int(pred[~pos].sum())
        acc = float((pred.float() == y).float().mean())
        model.train()
        return {"val_recall": recall, "val_fp": n_fp, "val_acc": acc}

    @torch.no_grad()
    def eval_fp_per_hour(self, model: WakeWordNet, chunk: int = 16384) -> dict[str, float]:
        """在 fp_val 特征流上滑 16 帧窗口（步长 1 帧）。

        fp_windows：分数 ≥ 0.5 的窗口数；activations：上升沿数（更接近实际的误唤醒次数）。
        """
        ds = self.ds
        if ds.fp_val is None:
            return {}
        model.eval()
        n = ds.n_frames
        T = len(ds.fp_val)
        fp_windows = 0
        activations = 0
        prev_active = False
        for start in range(0, T - n + 1, chunk):
            stop = min(T, start + chunk + n - 1)
            block = np.asarray(ds.fp_val[start:stop], dtype=np.float32)
            win = np.lib.stride_tricks.sliding_window_view(block, n, axis=0)  # (m, 96, 16)
            x = torch.from_numpy(np.ascontiguousarray(np.transpose(win, (0, 2, 1))))
            p = (model(x).squeeze(1) >= THRESHOLD).numpy()
            fp_windows += int(p.sum())
            edges = np.concatenate([[prev_active], p])
            activations += int(((~edges[:-1]) & edges[1:]).sum())
            prev_active = bool(p[-1])
        model.train()
        hours = ds.fp_val_hours
        return {"fp_windows_per_hour": fp_windows / hours, "activations_per_hour": activations / hours}

    def evaluate(self, model: WakeWordNet) -> dict[str, float]:
        m = self.eval_balanced(model)
        m.update(self.eval_fp_per_hour(model))
        return m

    # ------------------------------------------------------------ 训练
    def run_stage(
        self, stage: int, steps: int, lr: float, max_neg_weight: float, val_steps: set[int]
    ) -> None:
        from tqdm import tqdm

        model, opt = self.model, self.optimizer
        model.train()
        weights = np.linspace(1.0, max_neg_weight, steps)
        warmup, hold = steps // 5, steps // 3
        bar = tqdm(range(steps), desc=f"stage {stage} (lr={lr:g}, w≤{max_neg_weight:g})")
        for step in bar:
            for g in opt.param_groups:
                g["lr"] = lr_warmup_cosine(step, steps, warmup, hold, lr)
            x, y = self.batch()
            opt.zero_grad()
            p = model(x).squeeze(1)
            # 只对还没学会的样本回传
            keep = ((y == 0) & (p >= 1e-3)) | ((y == 1) & (p < 1 - 1e-3))
            if int(keep.sum()) < 32:
                keep = torch.ones_like(keep)
            w = torch.where(y == 1, torch.ones_like(y), torch.full_like(y, float(weights[step])))
            loss = torch.nn.functional.binary_cross_entropy(p[keep], y[keep], weight=w[keep])
            loss.backward()
            opt.step()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")
            if step in val_steps:
                metrics = self.evaluate(model)
                metrics.update({"stage": stage, "step": step, "loss": float(loss.item())})
                self.history.append(metrics)
                self.checkpoints.append(Checkpoint(step, stage, copy.deepcopy(model.state_dict()), metrics))
                shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
                bar.write(f"stage {stage} step {step}: {shown}")  # 用 tqdm 输出，不和进度条串行

    def train(self) -> tuple[WakeWordNet, dict]:
        t = self.cfg.train
        steps, lr, w = int(t.steps), float(t.lr), float(t.max_negative_weight)
        fp_key = "activations_per_hour"

        def best_fp() -> float:
            vals = [c.metrics.get(fp_key) for c in self.checkpoints if fp_key in c.metrics]
            return min(vals) if vals else 0.0

        # 第一段：后 25% 里取 20 个验证点；后两段：全程 20 个
        self.run_stage(1, steps, lr, w, set(np.linspace(steps * 0.75, steps - 1, 20).astype(int)))
        for stage in (2, 3):
            if self.checkpoints and best_fp() > t.target_fp_per_hour:
                w *= 2
                log.info(
                    "false positives above target (%.2f/h > %.2f/h): negative weight → %g",
                    best_fp(),
                    t.target_fp_per_hour,
                    w,
                )
            lr /= 10
            s = max(200, steps // 10)
            self.run_stage(stage, s, lr, w, set(np.linspace(1, s - 1, 20).astype(int)))

        final, info = self.select_final()
        return final, info

    # ------------------------------------------------------------ 选模型
    def select_final(self) -> tuple[WakeWordNet, dict]:
        """在误唤醒不高于目标的检查点里挑召回最高的几个做权重平均；平均后不如单个最好的就用单个。"""
        t = self.cfg.train
        cks = self.checkpoints or [
            Checkpoint(-1, 0, copy.deepcopy(self.model.state_dict()), self.evaluate(self.model))
        ]
        fp_key = "activations_per_hour" if any("activations_per_hour" in c.metrics for c in cks) else "val_fp"

        def score(c: Checkpoint) -> tuple:
            fp = c.metrics.get(fp_key, 0.0)
            return (0 if fp <= t.target_fp_per_hour else 1, -c.metrics["val_recall"], fp)

        ranked = sorted(cks, key=score)
        best = ranked[0]
        top = [c for c in ranked[:5] if score(c)[0] == score(best)[0]]
        candidates: list[tuple[str, dict[str, torch.Tensor], dict[str, float]]] = [
            (f"single@{best.stage}/{best.step}", best.state, best.metrics)
        ]
        if len(top) > 1:
            avg = average_state_dicts([c.state for c in top])
            m = WakeWordNet(self.ds.n_frames, self.ds.n_features, t.layer_size, t.n_blocks, t.dropout)
            m.load_state_dict(avg)
            candidates.append((f"average_of_{len(top)}", avg, self.evaluate(m)))

        def cand_score(c) -> tuple:
            fp = c[2].get(fp_key, 0.0)
            return (0 if fp <= t.target_fp_per_hour else 1, -c[2]["val_recall"], fp)

        name, state, metrics = sorted(candidates, key=cand_score)[0]
        final = WakeWordNet(self.ds.n_frames, self.ds.n_features, t.layer_size, t.n_blocks, t.dropout)
        final.load_state_dict(state)
        final.eval()
        log.info(
            "selected %s: %s", name, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
        )
        return final, {
            "selected": name,
            "metrics": metrics,
            "candidates": [{"name": n, "metrics": m} for n, _, m in candidates],
        }


def run_training(cfg: TrainingConfig, precomputed: Path | None, validation: Path | None) -> Path:
    ds = load_datasets(cfg, precomputed, validation)
    trainer = Trainer(cfg, ds)
    t0 = time.time()
    model, info = trainer.train()
    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, cfg.export_dir / f"{cfg.model_name}.onnx")
    torch.save(model.state_dict(), cfg.export_dir / f"{cfg.model_name}.pt")
    meta = {
        "model_name": cfg.model_name,
        "wake_phrase": cfg.wake_phrase,
        "language": "yue (Cantonese)",
        "input_shape": [ds.n_frames, ds.n_features],
        "framework": "openwakeword",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_seconds": round(time.time() - t0, 1),
        "data": {
            "positive_train": int(len(ds.pos_train)),
            "negative_train": int(len(ds.neg_train)),
            "positive_val": int(len(ds.pos_val)),
            "negative_val": int(len(ds.neg_val)),
            "real_recordings": dict(ds.real),  # 上面四个数里来自真人录音的部分
            "precomputed_negatives": 0 if ds.precomputed is None else int(len(ds.precomputed)),
            "fp_validation_hours": round(ds.fp_val_hours, 2),
        },
        "selection": info,
        "config": cfg.to_dict(),
    }
    (cfg.export_dir / f"{cfg.model_name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (cfg.export_dir / "history.json").write_text(json.dumps(trainer.history, indent=2))
    log.info("exported %s", onnx_path)
    return onnx_path
