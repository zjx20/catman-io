# 粤语唤醒词训练

把「小貓人」（或任何粤语短语）训练成 openWakeWord 可用的唤醒词模型。整个流程在开发机上跑，
CPU 即可（不需要 GPU）；目标机器只需要导出的 `.onnx`。

## 流程

```
synth      edge-tts 把文本清单展开成 16 kHz WAV       →  work/<name>/clips/*.wav + manifest.jsonl
resources  下载 MIT 房间冲激响应、openWakeWord 验证集特征、通用负样本特征切片
features   增强（变速/混响/噪声/滤波/失真/音量）→ openWakeWord 嵌入特征  →  work/<name>/features/*.npy
train      训练小分类器、按每小时误唤醒挑检查点、导出 ONNX          →  work/<name>/export/<name>.onnx
evaluate   在验证集音频上流式跑一遍，给出各阈值下的召回率 / 误接受率 / 每小时误唤醒
install    把 ONNX 和元数据复制到 catman_io/wakeword/models/
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train]"

python -m training.wakeword all --config training/wakeword/configs/siu_maau_jan.yaml
# 或分步：synth / resources / features / train / evaluate / install
python -m training.wakeword install --config training/wakeword/configs/siu_maau_jan.yaml
```

每一步都是幂等的：合成只补缺失的片段，下载支持断点续传，特征文件存在就跳过（`--overwrite` 重算）。
默认配置合成约 2800 条片段（约 8 分钟，取决于网络），特征几分钟，训练在 4 核 CPU 上约 5～10 分钟。
`evaluate` 除了各阈值的召回率 / 误接受率，还会按负样本类别（adversarial / general）统计误接受，
并列出最容易被误接受的短语——想压误唤醒时，先看这一行。

## 数据从哪来

| 数据 | 来源 | 说明 |
|---|---|---|
| 正样本 | edge-tts zh-HK 三个音色 × 5 档语速 × 5 档音高 × 8 种读法（标点 / 前缀） | 600 条 |
| 对抗负样本 | `phrases.py` 手写：只说一半、换字、换调、换序、相似常用词 | 约 90 个短语 × 3 音色 × 3 语速 |
| 普通负样本 | 130 句日常粤语 + 46 句普通话（zh-CN 音色）+ 20 句英语（en-US 音色） | |
| 通用负样本特征 | openWakeWord 作者预计算的 ACAV100M 特征，按小时只下载开头一段（每小时约 15 MB） | `precomputed_negative_hours`，默认 100 小时 |
| 误唤醒验证集 | openWakeWord 的 11 小时验证集特征（约 180 MB） | `fp_validation` |
| 房间混响 | MIT 环境冲激响应 270 条（约 8 MB） | `download_rirs` |
| 环境噪声 | **需要自己提供**（`background_dirs`），没有就只用合成噪声 / 人声嘈杂 | 强烈建议 |

正样本在训练时**右对齐**放进 2 秒窗口——唤醒词刚说完的那一刻就是模型应当触发的时刻，
这也是为什么 `phrases.py` 里的正样本只允许前缀（"喂，小貓人"）不允许后缀。

## 让模型在真实环境里好用

合成语音只有三个说话人，真实场景的差距主要来自说话人、麦克风和房间。按收益排序：

1. **真人录音正样本**：用目标设备录几十条不同人、不同距离、不同语气的「小貓人」，
   放到一个目录，配到 `data.extra_positive_dirs`。真录音会以 3 倍轮数参与增强。
2. **真实环境噪声**：用目标设备录几段电视、厨房、街声、空调，配到 `augment.background_dirs`。
3. **误唤醒录音**：把日常对话、电视声录下来放到 `data.extra_negative_dirs`；
   如果线上出现了具体的误唤醒句子，把那句话加进 `data.adversarial_phrases`。
4. 调阈值：`evaluate` 会打印各阈值下的召回率和每小时误唤醒，按需要在 `catman-io wake -t` 里调。

## 已经试过的配置

| 配置 | 合成验证集召回@0.5（干净 / 加噪） | 日常句子误接受 | 通用音频误唤醒 @0.5 / @0.9 |
|---|---|---|---|
| 20 h 通用负样本、增强 4/2 轮、20000 步、无 dropout | 100% / 98.9% | 0.6% | 38 次/h / 25 次/h（死记训练负样本） |
| **100 h、增强 8/4 轮、30000 步、dropout 0.1（随包 v0）** | 94.7% / 91.5% | 0% | 0.19 次/h / 0 |
| 同上，分类器加宽到 64 | 95.7% / 94.7% | 0.4% | 0.19 次/h / 0.19 次/h |

加宽分类器只多换来两三个点的加噪召回，对抗短语误接受却从 8.7% 升到 12.2%，而且高阈值下误唤醒不再归零，
所以默认保持 32。想提升召回，加真人录音比调结构有效得多。

## 换一个唤醒词

复制 `configs/siu_maau_jan.yaml`，改 `model_name` / `wake_phrase` / `workdir`，
把 `data.positive_phrases` 填成新唤醒词的各种读法，`data.adversarial_phrases` 按粤拼写一批
相近短语（只说一半、换一个字、同音不同调），其余可以沿用默认清单。

## 目录结构

```
training/wakeword/
  configs/siu_maau_jan.yaml   训练配置
  phrases.py                  文本清单（正样本读法、对抗负样本、日常句子）
  config.py                   配置 dataclass（所有键的默认值在这里）
  tts.py                      edge-tts 合成、解码、裁静音、manifest
  resources.py                下载 RIR / 验证集特征 / 通用负样本特征切片（HTTP Range）
  augment.py                  数据增强（numpy / scipy）
  features.py                 增强 + openWakeWord 嵌入特征 → memmap .npy
  model.py                    分类器（与 openWakeWord "dnn" 一致）+ ONNX 导出
  train.py                    训练循环（改写自 openWakeWord train.py）
  evaluate.py                 评估
  __main__.py                 命令行
  work/                       训练产物（不进仓库）
```

训练策略改写自 openWakeWord 的 `train.py`（Apache-2.0，David Scripka），特征提取直接调用
`openwakeword.utils.AudioFeatures`；不用它的 `data.py`，是因为那条路依赖 speechbrain /
torch-audiomentations 和只支持英语的音素对抗样本生成。
