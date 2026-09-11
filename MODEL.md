# 唤醒词模型 v0（小貓人）

模型名 `siu_maau_jan`（openWakeWord 分类器）。这条分支 = 训练它的代码提交 7f6b08d4cf53 + 本提交（模型、训练数据、配置），由 `scripts/wakeword_model.py publish` 生成。

- 发布时间：2026-09-11T02:15:47Z
- 训练代码：commit 7f6b08d4cf53（分支 （不在任何本地分支上））
- 训练时间：2026-09-08T18:04:16Z，953.0 s
- 训练数据：正样本 4048 / 负样本 7416（增强后），通用负样本 281250 段，误唤醒验证集 10.7 h
- 合成语速：正样本 -20% -10% +0% +10% +20%；负样本 -10% +0% +15%；音色 3 个
- 后处理变速：重采样变速 p=0.5 [0.9, 1.0, 1.1]
- 分类器：32 宽 × 1 块，dropout 0.1，30000 步，负样本权重上限 1500.0，误唤醒目标 0.2/h
- 召回@0.5：干净 94.7%，增强 91.5%
- 误接受@0.5：adversarial 8.7%，general 0.0%
- 通用音频误唤醒：0.187/h @0.5，0.0/h @0.9（10.7 h）

## 说明

随包的第一版。正样本语速 -20% 到 +20%，重采样变速 0.9～1.1。真机：慢速和正常语速可用，说快了分数有起伏但到不了阈值。训练片段当时没有归档，这条分支不带；配置 YAML 按模型元数据里记录的实际配置还原（当时仓库里的 YAML 已改成 +30%/+40% 但没有重训）。在 v1/v2 的验证集（语速到 +100%）上重新评估：干净召回 88.0%，1.25/1.5/1.75/2 倍速 80.0/60.0/36.0/24.0%，对抗短语误接受 10.6%，通用音频误唤醒 0.19/h。

## 本提交带的文件

- `catman_io/wakeword/models/siu_maau_jan.json`
- `catman_io/wakeword/models/siu_maau_jan.onnx`
- `training/wakeword/configs/siu_maau_jan.yaml`
- `training/wakeword/work/siu_maau_jan/export/siu_maau_jan.json`
- `training/wakeword/work/siu_maau_jan/export/siu_maau_jan.onnx`
- `MODEL.md`

## 使用

```bash
python scripts/wakeword_model.py pull v0     # 只取模型文件到 catman_io/wakeword/models/
git checkout models/wakeword/v0                # 要复现 / 重训：代码、配置、合成片段都在
python -m training.wakeword resources --config training/wakeword/configs/siu_maau_jan.yaml
```
