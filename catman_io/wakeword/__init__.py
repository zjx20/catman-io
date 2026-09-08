"""唤醒词检测：openWakeWord 特征流水线 + 自训练的粤语「小貓人」分类器。"""

from .detector import Detection, WakeWordDetector, bundled_models, default_model_path, ensure_base_models

__all__ = ["Detection", "WakeWordDetector", "bundled_models", "default_model_path", "ensure_base_models"]
