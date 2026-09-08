import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.wakeword.model import (  # noqa: E402
    WakeWordNet,
    average_state_dicts,
    export_onnx,
    onnx_predictor,
)
from training.wakeword.train import lr_warmup_cosine  # noqa: E402


def test_net_shapes_and_onnx_export(tmp_path):
    net = WakeWordNet(16, 96, layer_size=8, n_blocks=1).eval()
    x = torch.rand(3, 16, 96)
    y = net(x)
    assert y.shape == (3, 1) and (0 <= y).all() and (y <= 1).all()
    path = export_onnx(net, tmp_path / "m.onnx")
    pred = onnx_predictor(path)
    got = pred(x.numpy())
    assert got.shape == (3,)
    assert np.allclose(got, y.detach().numpy().reshape(-1), atol=1e-4)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].shape[1:] == [16, 96]  # openwakeword 读 shape[1] 当帧数


def test_average_state_dicts():
    a, b = WakeWordNet(layer_size=4), WakeWordNet(layer_size=4)
    avg = average_state_dicts([a.state_dict(), b.state_dict()])
    for k in avg:
        assert torch.allclose(avg[k], (a.state_dict()[k] + b.state_dict()[k]) / 2)


def test_lr_schedule():
    assert lr_warmup_cosine(0, 100, 20, 30, 1e-3) == 0
    assert lr_warmup_cosine(20, 100, 20, 30, 1e-3) == pytest.approx(1e-3)
    assert lr_warmup_cosine(40, 100, 20, 30, 1e-3) == pytest.approx(1e-3)
    assert lr_warmup_cosine(99, 100, 20, 30, 1e-3) < 1e-5


def test_trainer_learns_separable_data(tmp_path):
    """两簇随机特征，几百步就该分开——用来守住训练循环本身没写坏。"""
    from training.wakeword.config import TrainingConfig
    from training.wakeword.train import Datasets, Trainer

    rng = np.random.default_rng(0)
    pos = rng.normal(0.5, 1.0, (200, 16, 96)).astype(np.float32)
    neg = rng.normal(-0.5, 1.0, (200, 16, 96)).astype(np.float32)
    ds = Datasets(pos[:150], neg[:150], pos[150:], neg[150:])
    cfg = TrainingConfig(workdir=str(tmp_path))
    cfg.train.steps = 300
    cfg.train.batch_positive = cfg.train.batch_negative = 16
    cfg.train.batch_precomputed = 0
    cfg.train.layer_size = 8
    cfg.train.max_negative_weight = 5
    cfg.train.threads = 1
    trainer = Trainer(cfg, ds)
    model, info = trainer.train()
    m = trainer.eval_balanced(model)
    assert m["val_recall"] > 0.9 and m["val_fp"] <= 5, m
