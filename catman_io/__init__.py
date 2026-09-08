"""catman-io：catman 的语音输入输出软件栈（粤语优先）。

模块一览（按音频从麦克风到后端的流向）：

- :mod:`catman_io.audio`    采集 / 播放 / 帧工具（16 kHz、80 ms 一帧）
- :mod:`catman_io.wakeword` 唤醒词检测（openWakeWord + 自训练的粤语模型）
- :mod:`catman_io.vad`      语音活动检测（规划中）
- :mod:`catman_io.asr`      粤语语音转文字（规划中）
- :mod:`catman_io.intent`   意图识别（规划中）
- :mod:`catman_io.stream`   音频流推送到后端 live 模型（规划中）
- :mod:`catman_io.tts`      粤语语音合成 / 音频输出（规划中）
"""

__version__ = "0.1.0"
