# CLAUDE.md

面向在本仓库开发 catman-io 的 Claude Code / 开发者。

## 这是什么

catman（https://github.com/zjx20/catman）的语音输入输出组件：粤语优先的语音助手软件栈，
跑在一台只有 CPU 的小主机上，音频设备自带远场拾音与降噪。Python 3.10/3.11。
目前完成：音频采集 / 帧约定、粤语唤醒词「小貓人」检测与训练流水线；
VAD、ASR、意图、推流、TTS 只有接口占位（见 README 路线图）。

## 常用命令

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 训练 / 测试需要，用 CPU 版
pip install -e ".[dev,train,audio]"
catman-io setup                      # 下载 openWakeWord 基础模型（约 5 MB，tests 也需要）
ruff check catman_io training tests && pytest -q

catman-io wake-file tests/data/positive_siu_maau_jan_hiugaai.wav --scores   # 离线跑一条录音
python -m training.wakeword all --config training/wakeword/configs/siu_maau_jan.yaml
python -m training.wakeword install --config training/wakeword/configs/siu_maau_jan.yaml
```

## 约定

- 全栈音频统一 **16 kHz、单声道、int16，80 ms（1280 采样点）一帧**，常量在 `catman_io/audio/frames.py`。
- `catman_io/` 是装到目标机器的运行时包，依赖刻意很轻（numpy、openwakeword、onnxruntime、pyyaml）；
  `sounddevice` 是可选的 `[audio]`，torch / edge-tts 等只在 `[train]`。别把训练依赖引进运行时。
- `training/` 不随包安装，用 `python -m training.wakeword ...` 从仓库根目录运行；产物在
  `training/wakeword/work/`（gitignore），最终模型手动 `install` 到 `catman_io/wakeword/models/`。
- 唤醒词模型文件名就是 openWakeWord 里的模型名（`siu_maau_jan.onnx` → 检测结果里的 `model="siu_maau_jan"`），
  旁边的同名 `.json` 记录训练数据、评估结果和配置。
- 正样本训练时右对齐到 2 秒窗口末尾，所以 `phrases.py` 里正样本只能加前缀不能加后缀；
  负样本文本绝不能包含唤醒词（`tests/test_phrases.py` 守着）。
- 文档、注释、提交信息用中文；README 面向使用者，不提具体硬件型号。

## 排查

- `catman-io devices` 报 PortAudio not found：`apt install libportaudio2`。
- edge-tts 在代理 / 自签 CA 环境下证书报错：设置 `SSL_CERT_FILE` 指向 CA bundle，`tts.py` 会尊重它。
- Python 3.12+ 装不上 openwakeword：是它依赖的 tflite-runtime 没有轮子，用 3.11。
