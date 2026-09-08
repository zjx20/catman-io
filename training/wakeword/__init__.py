"""粤语唤醒词训练流水线：edge-tts 合成 → 数据增强 → openWakeWord 特征 → 训练小分类器 → 导出 ONNX。

用法（在仓库根目录）::

    python -m training.wakeword all --config training/wakeword/configs/siu_maau_jan.yaml

各步骤也可以单独跑：synth / resources / features / train / evaluate / install。
"""
