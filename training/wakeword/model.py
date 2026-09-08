"""分类器结构与 openWakeWord 的 "dnn" 模型一致：
Flatten → Linear → LayerNorm → ReLU → (block)×n → Linear → Sigmoid。

输入 (batch, 16, 96) 的嵌入特征，输出 (batch, 1) 的唤醒概率。导出的 ONNX 可以直接交给
openwakeword.Model(wakeword_models=[...], inference_framework="onnx") 使用。
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn


class WakeWordNet(nn.Module):
    def __init__(self, n_frames: int = 16, n_features: int = 96, layer_size: int = 32, n_blocks: int = 1):
        super().__init__()
        self.n_frames, self.n_features = n_frames, n_features
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(n_frames * n_features, layer_size)
        self.norm1 = nn.LayerNorm(layer_size)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_size, layer_size), nn.LayerNorm(layer_size), nn.ReLU())
                for _ in range(n_blocks)
            ]
        )
        self.out = nn.Linear(layer_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.out(x))


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {k: v.detach().clone().float() for k, v in states[0].items()}
    for sd in states[1:]:
        for k in out:
            out[k] += sd[k].float()
    for k in out:
        out[k] /= len(states)
    return out


def export_onnx(model: WakeWordNet, path: Path) -> Path:
    """导出 ONNX（batch 维动态，帧数固定），并用 onnxruntime 复核输出一致。"""
    m = copy.deepcopy(model).eval().cpu()
    dummy = torch.rand(1, m.n_frames, m.n_features)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(m, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # 老版本 torch 没有 dynamo 参数
        torch.onnx.export(m, dummy, str(path), **kwargs)

    x = np.random.rand(4, m.n_frames, m.n_features).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = onnx_predictor(path)(x)[:, None]
    if not np.allclose(ref, got, atol=1e-4):
        raise RuntimeError(f"ONNX export mismatch: max diff {np.abs(ref - got).max()}")
    return path


def onnx_predictor(path: Path, threads: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = threads
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def predict(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {name: x.astype(np.float32)})[0].reshape(-1)

    return predict
