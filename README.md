# catman-io

[catman](https://github.com/zjx20/catman) 的**语音输入输出**组件：一套**以粤语为前提**的智能语音助手软件栈。
它面向已经做好远场拾音和降噪的音频设备，负责后面的一切——把处理过的音频变成"唤醒 → 听懂 → 回应"，
运行在一台小主机上，只用 CPU。

```
麦克风/拾音设备 ──► 采集(16 kHz · 80 ms 一帧)
                      │
                      ▼
                 唤醒词检测「小貓人」 ◄── 本仓库当前完成的部分
                      │ 唤醒
                      ▼
            VAD 端点检测 ──► 粤语语音识别 ──► 意图识别 ──► catman 后端
                      │                                      │
                      └──► 音频流推送（live 模型）             ▼
                                                   粤语语音合成 ──► 扬声器
```

| 模块 | 职责 | 状态 |
|---|---|---|
| `catman_io.audio` | 采集 / 播放 / 帧工具，全栈统一 16 kHz 单声道、80 ms 一帧 | ✅ |
| `catman_io.wakeword` | 粤语唤醒词「小貓人」检测（openWakeWord + 自训练模型） | ✅ 可用，持续改进 |
| `training/wakeword` | 唤醒词训练流水线：edge-tts 合成 → 增强 → 训练 → 导出 ONNX | ✅ |
| `catman_io.vad` | 语音活动检测 / 端点检测 | 🚧 接口已定 |
| `catman_io.asr` | 粤语语音转文字 | 🚧 接口已定 |
| `catman_io.intent` | 意图识别（本地快捷指令 / 交给后端） | 🚧 接口已定 |
| `catman_io.stream` | 把音频流推给后端 live 模型 | 🚧 接口已定 |
| `catman_io.tts` | 粤语语音合成与音频输出 | 🚧 接口已定 |

## 唤醒词：「小貓人」

唤醒词是粤语的**「小貓人」**（siu2 maau1 jan4，简体写作"小猫人"）。检测基于
[openWakeWord](https://github.com/dscripka/openWakeWord)：它的 melspectrogram + 语音嵌入模型把音频变成
每 80 ms 一帧的 96 维特征（约 2.4 MB，两个 ONNX），我们只训练最后那个几十 KB 的小分类器。
整条链路在一个 CPU 核上是实时的，占用极低。

openWakeWord 官方的训练流程只支持英语（靠 Piper 合成样本、靠英语音素表造对抗样本），
本仓库把它改成了粤语版：

- **正样本**用 [edge-tts](https://github.com/rany2/edge-tts) 的三个粤语音色（zh-HK）合成，
  按语速 × 音高 × 标点/前缀展开成数百条不同读法；
- **对抗负样本**按粤拼手写：只说一半（小貓 / 貓人）、换一个字（小狗人、小貓神、燒貓人）、
  换序、常用相似词（小朋友、小籠包）等，见 `training/wakeword/phrases.py`；
- **普通负样本**是日常粤语句子，外加少量普通话和英语；再混入 openWakeWord 预计算的通用音频负样本
  （按需只下载十几小时那一小段，不用拉整个 17 GB）；
- **增强**：变速、房间混响（MIT RIR）、环境噪声、人声嘈杂、有色噪声、滤波、失真、音量——纯 numpy/scipy；
- **训练**沿用 openWakeWord 的策略（负样本权重线性上升、只对难样本回传、按每小时误唤醒挑检查点、权重平均）。

### 随包模型 v0 的成绩

`catman_io/wakeword/models/siu_maau_jan.onnx`（约 200 KB）只用合成语音训练，
下面是它在**合成**验证集和 10.7 小时通用音频上的表现（详见旁边的 `siu_maau_jan.json`）：

| 指标 | 阈值 0.5 | 阈值 0.7 |
|---|---|---|
| 召回率（干净合成语音 / 加噪加混响后） | 94.7% / 91.5% | 92.5% / 89.4% |
| 误接受：日常句子（粤 / 普 / 英） | 0.0% | 0.0% |
| 误接受：对抗短语（燒貓人、笑貓人、小貓銀 这类只差一个声调的） | 8.7% | 7.5% |
| 通用音频每小时误唤醒 | 0.19 次 | 0.09 次（0.9 时为 0） |

真人、真麦克风、真房间的效果会打折扣，这是所有纯合成训练的通病；补救办法见下文"训练 / 重训唤醒词"。

## 安装

需要 Python 3.10 或 3.11（openWakeWord 依赖的 tflite-runtime 目前没有更高版本的轮子）。

```bash
pip install -e ".[audio]"        # 运行时；audio = sounddevice，系统上要有 PortAudio（apt install libportaudio2）
catman-io setup                  # 下载 openWakeWord 的基础模型（约 5 MB），并列出随包的唤醒词模型
```

## 试一下

```bash
catman-io devices                                   # 看看拾音设备是哪个
catman-io wake -d <设备编号或名字>                    # 实时检测，对着麦克风说「小貓人」
catman-io wake-file recording.wav --scores          # 对一段录音离线检测，逐帧打印分数
```

阈值、冷却时间、连续帧数等都可以用参数或 YAML 配置（见 `config.example.yaml`）。

### 网页版测试（浏览器麦克风）

不想折腾 PortAudio，或者想边看分数曲线边试，可以起一个本地网页：

```bash
pip install -e ".[demo]"
catman-io webdemo --open          # 默认 http://127.0.0.1:8765 ，--record-dir 指定录音目录
```

页面里点"开始监听"，对着麦克风说「小貓人」，就能看到实时分数、命中记录、折合每小时的命中次数，
阈值 / 连续帧 / 冷却都能拖着调。**"保存最近 3 秒"按钮会把录音存成 16 kHz WAV**（默认 `data/recordings/`），
正好用来收集真人正样本和误唤醒片段，填进训练配置的 `extra_positive_dirs` / `extra_negative_dirs` 重训。
浏览器只允许 `http://localhost` 或 https 页面用麦克风，要在别的机器上开页面就走 SSH 隧道
（`ssh -L 8765:127.0.0.1:8765 <主机>`）。
在代码里使用：

```python
from catman_io.audio.capture import MicCapture
from catman_io.wakeword import WakeWordDetector

detector = WakeWordDetector(threshold=0.5, cooldown=2.0)
with MicCapture() as mic:
    for frame in mic.frames():          # 每帧 80 ms、1280 个 int16 采样点
        for det in detector.process(frame):
            print("唤醒！", det.model, det.score)
```

## 训练 / 重训唤醒词

详见 [training/wakeword/README.md](training/wakeword/README.md)。一句话版：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train]"
python -m training.wakeword all --config training/wakeword/configs/siu_maau_jan.yaml
python -m training.wakeword install --config training/wakeword/configs/siu_maau_jan.yaml
```

随包附带的模型只用合成语音训练。要在真实环境里好用，**最有效的两件事**是：用目标设备录几十条真人说的
「小貓人」放进 `extra_positive_dirs`，录几段房间噪声放进 `background_dirs`，然后重训。

## 开发

```bash
pip install -e ".[dev,train]"
ruff check catman_io training tests
pytest -q
```

## 路线图

1. ✅ 唤醒词检测（粤语）
2. VAD 端点检测：复用 openWakeWord 自带的 silero-vad
3. 粤语 ASR：评估 SenseVoice / Whisper / FunASR 粤语模型在 CPU 上的表现
4. 音频流推送：唤醒后把帧推给后端 live 模型，接收回传
5. 意图识别：本地快捷指令与后端分流
6. 粤语 TTS 与播放：先用 edge-tts zh-HK 音色，再评估离线方案
