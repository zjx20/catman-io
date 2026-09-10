# CLAUDE.md

面向在本仓库开发 catman-io 的 Claude Code / 开发者。

## 这是什么

catman（https://github.com/zjx20/catman）的语音输入输出组件：粤语优先的语音助手软件栈，
跑在一台只有 CPU 的小主机上，音频设备自带远场拾音与降噪。Python 3.10/3.11。
整条流水线已经能跑：唤醒词「小貓人」→ silero 端点检测 → sherpa-onnx 粤语识别 → 三层意图
（YAML 规则 → LLM function calling → 对话模型 / delegate 给 catman）→ 动作 → edge-tts 合成 → 扬声器；
每回合写日志、自动标 bad case，本机 HTTP API 让 catman 改规则并热加载。`catman_io.stream`（推流给 live 模型）
只有接口。

## 常用命令

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 训练 / 测试需要，用 CPU 版
pip install -e ".[dev,train,audio,asr,tts,demo]"
catman-io setup [--asr]              # 下载 openWakeWord 基础模型（tests 需要）；--asr 再下识别模型（170 MB）
ruff check catman_io training tests && pytest -q
CATMAN_IO_NETWORK_TESTS=1 pytest -q -m network                   # 联网测试（edge-tts），默认跳过
CATMAN_IO_TEST_ASR_ROOT=data/models/asr pytest -q tests/test_asr.py  # 有识别模型时跑真识别

catman-io run -c config.yaml                          # 整条流水线；--wav in.wav --out out.wav 离线跑
catman-io listen --wav tests/data/xxx.wav --no-asr    # 只跑唤醒 → 切句
catman-io intent parse "聽日會唔會落雨" / intent test / intent lint
catman-io journal list --bad / journal review
python -m training.wakeword all --config training/wakeword/configs/siu_maau_jan.yaml
```

## 约定

- 全栈音频统一 **16 kHz、单声道、int16，80 ms（1280 采样点）一帧**，常量在 `catman_io/audio/frames.py`。
- `catman_io/` 是装到目标机器的运行时包，核心依赖刻意很轻（numpy、openwakeword、onnxruntime、pyyaml、zhconv）；
  其余都是可选 extras：`audio`（sounddevice）、`asr`（sherpa-onnx）、`tts`（edge-tts、soundfile）、`demo`（aiohttp，
  网页 demo 与 HTTP API 都用它）；torch 只在 `[train]`。别把训练依赖引进运行时；新的重依赖放 extras 并在缺失时
  降级或给出安装提示。
- `training/` 不随包安装，用 `python -m training.wakeword ...` 从仓库根目录运行；产物在 `training/wakeword/work/`
  （gitignore），最终模型手动 `install` 到 `catman_io/wakeword/models/`。
- 唤醒词模型文件名就是 openWakeWord 里的模型名（`siu_maau_jan.onnx` → `Detection.model="siu_maau_jan"`），
  旁边同名 `.json` 记录训练数据、评估结果和配置。正样本训练时右对齐到 2 秒窗口末尾，所以 `phrases.py` 里
  正样本只能加前缀不能加后缀；负样本文本绝不能包含唤醒词（`tests/test_phrases.py` 守着）。
  快语速靠 TTS 合成到 +100%；`augment.time_stretch`（WSOLA 变速不变调）只用在 `evaluate` 的快语速压力测试里，
  训练时 `p_tempo` 保持 0（实测拿它增强训练集会拖低正常语速的召回）。
- **线程模型**（`pipeline.py`）：主线程跑帧循环，每帧过唤醒检测与 VAD，喂 `dialog.Dialog`（纯状态机，只有主线程碰）
  并执行它吐出的命令；worker 线程做识别 / 应答 / 合成，通过事件队列汇报；Speaker 自带写线程；API 在自己的 loop 线程。
  回合有 `gen` 代号与 `cancelled` 标志，`speaker.say(pcm, gen)` 丢弃过期回合的音频——打断靠这个，别绕过它。
  `--wav` 模式时钟是"帧数 × 80 ms"，等 worker 时按真实节奏喂帧。
- **意图规则**：模式写在归一化后的文本上（`intent/normalize.py`：香港繁体、小写、无标点），`{槽位}` 展开成命名组，
  `{tail}` / `{polite}` 是宏；每个意图的 `examples` 必须全部命中本意图（`RuleSet.lint`），`negatives` 不得命中。
  内置规则在 `catman_io/intent/rules/builtin.yaml`，现场规则在 `<data_dir>/intent/rules/*.yaml`（同名覆盖，热加载）。
  改内置规则要同时跑 `catman-io intent lint` 与 `pytest tests/test_rules.py`。新意图会自动变成 LLM 层的工具。
- **日志字段**：`journal/__init__.py:build_record` 是唯一的字段来源；标记逻辑在 `journal/flags.py`，加新标记要同时加进
  `BAD_FLAGS` / `FLAG_HELP` 并补一个 `tests/test_journal.py` 用例。日志行的 `kind` 是行类型（turn / amend），回合类型叫 `turn_kind`。
- **HTTP API**（`api/server.py`）是给 catman 的合同：路由与语义写在模块顶部注释里，改了要同步
  `integrations/catman/skills/catman-io/SKILL.md`。PUT 规则必须过 lint 与全部用例才落盘。
- 配置：`config.py` 递归解析、未知键报错、密钥只经 `*_env`（`secret()` 用时读取）；新 section 要同步 `config.example.yaml`。
- 文档、注释、提交信息用中文；README 面向使用者，不提具体硬件型号。日志与异常信息用英文。
- 测试不需要麦克风 / 扬声器 / 网络：假识别、假合成、`ListOutput` 扬声器、注入的 HTTP transport；
  标 `network` 的默认跳过；真识别测试靠 `CATMAN_IO_TEST_ASR_ROOT`。`tests/test_webdemo.py` 有检查前后端消息类型一致的测试。

## 排查

- `catman-io devices` 报 PortAudio not found：`apt install libportaudio2`。
- edge-tts 在代理 / 自签 CA 环境下证书报错：设置 `SSL_CERT_FILE` 指向 CA bundle，`tts/edge.py` 与训练的 `tts.py` 都尊重它。
- Python 3.12+ 装不上 openwakeword：是它依赖的 tflite-runtime 没有轮子，用 3.11。
- `intent.llm.enabled` 打开但启动就报 `CATMAN_IO_LLM_API_KEY is not set`：故意的，密钥缺失要早失败。
- catman 在 Docker 里连不上 API：`api.host` 要是 `0.0.0.0`，容器里用 `http://host.docker.internal:8766`。
