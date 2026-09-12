# catman-io

[catman](https://github.com/zjx20/catman) 的**语音输入输出**组件：一套**以粤语为前提**的智能语音助手软件栈。
它面向已经做好远场拾音和降噪的音频设备，负责后面的一切——把处理过的音频变成"唤醒 → 听懂 → 回应"，
运行在一台只有 CPU 的小主机上。

目标是天猫精灵那种响应速度，加上大模型的智能：简单指令（報時、計時、天氣、音量……）走本地规则，
零延迟、可离线；说法灵活的指令交给一次 flash 级模型的 function calling（约一秒）；闲聊问答交给对话模型
流式回答；要动手做事的任务交给 catman。每一回合都记日志、自动标出 bad case，让 catman 定期把 bad case
变成更好的规则——**规则越用越准，不用人盯着**。

```
拾音设备 ─► 采集(16 kHz · 80 ms/帧) ─► 唤醒词「小貓人」 ─► 端点检测(silero VAD) ─► 粤语识别(SenseVoice)
                                                                                        │
              扬声器 ◄─ 粤语合成(edge-tts) ◄─ 应答 ◄─┬─ ① 规则（YAML，零延迟）◄──────────┘
                 ▲                                  ├─ ② LLM function calling（约 1 s）
                 │                                  ├─ ③ 对话模型（流式）
              提示音、打断、免唤醒跟进               └─ delegate → catman（结果走微信）
                                                            │
                                     回合日志 → 自动标记 bad case → HTTP API → catman 改规则 → 热加载
```

| 模块 | 职责 | 状态 |
|---|---|---|
| `catman_io.audio` | 采集 / 帧工具，全栈统一 16 kHz 单声道、80 ms 一帧 | ✅ |
| `catman_io.wakeword` + `training/wakeword` | 粤语唤醒词「小貓人」检测与训练流水线 | ✅ 可用，持续改进 |
| `catman_io.vad` | silero VAD + 端点检测（何时开口、何时说完） | ✅ |
| `catman_io.asr` | 粤语语音转文字（sherpa-onnx：SenseVoice / WenetSpeech-Yue） | ✅ |
| `catman_io.intent` | 三层意图：规则 → LLM → 落空；归一化、槽位、回归用例 | ✅ |
| `catman_io.actions` | 内置动作（報時 / 計時 / 音量 / 天氣…）与通用 http 动作 | ✅ |
| `catman_io.brain` | 对话模型（OpenAI 兼容端点，流式，短期记忆）；catman 客户端 | ✅ |
| `catman_io.tts` | edge-tts 粤语合成、可打断的扬声器、提示音 | ✅ |
| `catman_io.dialog` / `pipeline` | 对话状态机与整条流水线（`catman-io run`） | ✅ |
| `catman_io.journal` / `api` | 回合日志、bad case 标记、复盘包、给 catman 的 HTTP API | ✅ |
| `catman_io.webdemo` | 浏览器麦克风测唤醒词 | ✅ |
| `catman_io.stream` | 把音频流推给 live 模型 | 🚧 接口已定 |

## 快速开始

需要 Python 3.10 或 3.11（openWakeWord 依赖的 tflite-runtime 没有更高版本的轮子）。

```bash
pip install -e ".[audio,asr,tts,demo]"   # audio 要系统有 PortAudio（apt install libportaudio2）
catman-io setup --asr                    # 下载 openWakeWord 基础模型（5 MB）和粤语识别模型（约 170 MB）
python scripts/wakeword_model.py pull    # 从 models/wakeword/<版本> 分支取「小貓人」唤醒词模型（约 200 KB）
cp config.example.yaml config.yaml       # 按需改：设备、经纬度（天气）、LLM 端点……
export CATMAN_IO_LLM_API_KEY=...         # 开了 intent.llm / brain.llm 才需要
catman-io run -c config.yaml
```

对着麦克风说「小貓人」，听到提示音后说话：「而家幾點」「幫我計十分鐘」「聽日會唔會落雨」「大聲啲」，
或者随便聊。回答期间再叫「小貓人」可以打断；回答完几秒内不用叫唤醒词可以接着说。

不想插麦克风也能整条跑：`catman-io run --wav 录音.wav --out 回复.wav`。

## 命令一览

| 命令 | 用途 |
|---|---|
| `catman-io run` | 整条流水线（`--wav/--out` 用文件代替麦克风 / 扬声器；`--no-tts`、`--no-asr`、`--no-api`） |
| `catman-io listen` | 只跑前半段：唤醒 → 切句 → 存 WAV → 识别，在设备上调阈值用 |
| `catman-io wake` / `wake-file` / `webdemo` | 唤醒词：实时仪表 / 离线检测 / 浏览器页面 |
| `catman-io asr 录音.wav` | 试识别器（`--backend sensevoice\|wenet_yue`） |
| `catman-io say "你好"` | 试合成与扬声器（`-o out.wav` 写文件） |
| `catman-io intent parse "聽日會唔會落雨"` | 看一句话命中哪条规则（`--llm` 也问一次模型） |
| `catman-io intent test` / `lint` / `list` | 跑回归用例 / 检查规则 / 列意图 |
| `catman-io journal list --bad` / `show` / `review` | 看回合日志、bad case 复盘 |
| `catman-io flywheel export` / `nudge` / `ack` | 导出复盘、叫 catman 来处理、记游标 |
| `catman-io setup [--asr NAME] [--list-asr]` / `devices` | 模型与设备 |

## 它怎么响应得快

一句「查天气」从说完到听见第一个字，大头在端点检测的尾静音（0.6 s）、识别（2 核 CPU 上零点几秒）和
合成首包，不在意图判断。所以：

- 规则命中的回复**零延迟**，固定短语启动时预先合成进缓存；
- 只有规则没中才问 LLM，一次 function calling，超时（默认 4 s）就落到对话模型；
- 对话模型流式回答，第一句到了就开口；思考超过 1.5 s 先给一声提示音；
- 回答完后几秒是免唤醒的跟进窗口，播报中随时可用唤醒词打断。

三层意图：

| 层 | 做什么 | 延迟 | 何时用 |
|---|---|---|---|
| ① 规则 | `catman_io/intent/rules/builtin.yaml` + 现场 `data/intent/rules/*.yaml`，粤语归一化 + 槽位解析 | ~0 | 每句先过 |
| ② LLM | OpenAI 兼容端点一次 function calling，工具由规则文件生成 | ~1 s | 规则没中且配了 `intent.llm` |
| ③ 大脑 | `brain.llm` 对话模型流式回答；`delegate` 把任务交给 catman | 秒级 | LLM 判为闲聊 / 任务 |

内置意图：報時、日期、計時（含到点提醒）、音量、靜音、停、再講、幫助、天氣（Open-Meteo，免 key）。
新技能只改 YAML（`http` 动作能调任何 REST 接口，例如 Home Assistant），LLM 层自动拿到对应工具。

## 意图规则

```yaml
version: 1
intents:
  - name: timer.set
    description: 設定倒數計時
    slots: {duration: duration}               # 槽位类型：number / duration / time / date / percent / room / text
    examples: ["幫我set個十分鐘嘅timer", "三個字之後叫我"]   # lint 要求全部命中本意图
    patterns:
      - "^{polite}(?:set|較|校)(?:一)?個?{duration}(?:嘅)?(?:timer|鬧鐘){tail}$"
    action: builtin.timer
```

模式写在归一化后的文本上：无标点、小写、香港繁体（识别结果是简体也会先转繁），`{tail}` / `{polite}`
是语气词与客气话的宏，槽位解析照顾香港说法（三個字 = 15 分鐘、三點三 = 3:15、下星期三）。
现场规则放 `data/intent/rules/site.yaml`（同名意图覆盖内置），运行中改了会自动重新加载；
`catman-io intent test` 用 `data/intent/cases.jsonl` 的回归用例守着旧行为。

## 飞轮：bad case 自动收集，catman 来改

每回合写进 `data/journal/`（文本、意图、回复、延迟、音频），自动打标记：规则没中但 LLM 判出了意图、
用户马上重说或否定、误唤醒、识别为空、太慢……`catman-io run` 内嵌一个本机 HTTP API，catman 的助手按
[integrations/catman/skills/catman-io/SKILL.md](integrations/catman/skills/catman-io/SKILL.md) 定期
拉复盘、改 `site.yaml`、干跑回归、写回（自动 lint + 跑用例，过了才生效）。
说明见 [docs/flywheel.md](docs/flywheel.md)，接法见 [integrations/catman/README.md](integrations/catman/README.md)。

## 粤语识别

用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 跑离线 int8 模型，整句进文本出。
用 edge-tts 合成的 30 句粤语指令（三个音色）实测，2 线程：

| 模型（`catman-io setup --asr 名字`） | 字错率 | RTF | 说明 |
|---|---|---|---|
| `sense-voice-2025-09`（默认） | 4.3% | 0.058 | SenseVoice-Small，用 2.18 万小时粤语微调；不出标点 |
| `sense-voice-2024-07` | 3.4% | 0.058 | 原版，`use_itn` 可出标点 |
| `wenet-yue-2025-09` | 4.7% | 0.030 | 粤语专用 CTC，最小最快 |

错误集中在英文词（set / timer / catman）和语气词写法，归一化后不影响规则匹配。真人真环境会打折扣，
`catman-io listen` 可以在设备上边听边看识别结果。

## 唤醒词：「小貓人」

唤醒词是粤语的**「小貓人」**（siu2 maau1 jan4，简体写作"小猫人"）。检测基于
[openWakeWord](https://github.com/dscripka/openWakeWord)：它的 melspectrogram + 语音嵌入模型把音频变成
每 80 ms 一帧的 96 维特征（约 2.4 MB，两个 ONNX），我们只训练最后那个几十 KB 的小分类器。

openWakeWord 官方的训练流程只支持英语，本仓库把它改成了粤语版：正样本用 edge-tts 的三个粤语音色按
语速 × 音高 × 前缀展开；对抗负样本按粤拼手写（小貓 / 貓人 / 小狗人 / 燒貓人 / 小朋友…）；普通负样本是
日常粤语句子加少量普通话、英语，再混入 openWakeWord 预计算的通用音频负样本；增强用纯 numpy/scipy
（变速、混响、噪声、人声嘈杂、滤波、失真）；训练沿用 openWakeWord 的策略。详见
[training/wakeword/README.md](training/wakeword/README.md)。

**模型有版本，不放在代码分支里。** 每个版本一条 `models/wakeword/<版本>` 分支，里面是训练它的那份代码、
模型文件（`catman_io/wakeword/models/siu_maau_jan.onnx` + 同名 `.json`，约 200 KB）、训练用的合成片段与真人录音
和一份自动生成的说明 `MODEL.md`；`catman_io/wakeword/models/VERSION` 写着当前代码推荐的版本。

```bash
python scripts/wakeword_model.py list      # 有哪些版本
python scripts/wakeword_model.py pull      # 取推荐版本（或 pull v0 指定版本）到 catman_io/wakeword/models/
python scripts/wakeword_model.py show v2   # 看某个版本的训练配置、评估与真机备注
```

| 版本 | 正样本语速 | 后处理变速 / 真人录音 | 合成验证集召回@0.5（干净 / 1.5 倍速） | 真人录音召回@0.5 | 真机 |
|---|---|---|---|---|---|
| v0 | -20% ～ +20% | 重采样 0.9～1.1 | 88.0% / 60.0% | 71%（31 条） | 慢速、正常语速可用，说快了漏 |
| v1 | -20% ～ +100% | 重采样 0.85～1.3 | 92.6% / 88.6% | 58% | 中等语速比 v0 差，快语速没明显改善 |
| v2 | -20% ～ +100% | 无；近音短语不再作为负样本 | 100% / 89.1% | 58% | 不理想 |
| v3 | -20% ～ +20% | 重采样 0.9～1.1；真人录音 31 条（23 训练 / 8 验证，权重 8） | 【待填】 | 留出 8 条【待填】；四折交叉验证 100% | 待测 |

各版本日常句子误接受 ≤ 1%、通用音频每小时误唤醒 ≤ 0.19 次（阈值 0.5）。合成验证集上的数字并不能代替真机：
v1、v2 在合成集上全面领先 v0，真人录音上反而更差——把 TTS 语速加到 +100% 并没有换来真人快说的召回，只是把正常语速的
正样本稀释了。v2 解决的是另一件事：「小貓銀」「燒貓人」这类只差一个声调的近音负样本，小分类器分不开，硬压它们会把同一
语速档的正样本一起压掉，所以从 v2 起不再拿它们训练（代价是这类短语有约 78% 会触发）。v3 用 webdemo 收的 31 条真人录音
（两位说话人）进训练，是目前最有效的一步：没有真人录音的各种配方在这 31 条上只有 58%～71%，加进去之后按说话人分层的
四折交叉验证折外召回 100%，之前所有版本都给 0 分的几条（快说、轻声）全部上来了。它只证明了对这两位说话人泛化得好，
没录过的人靠合成语音兜底；**再录几个人、几种距离和语气**放进 `training/wakeword/real/siu_maau_jan/positive/<说话人>/`
重训，收益仍然最大；`catman-io run` 里每次唤醒前后那一段音频也会存进日志目录，误唤醒的可以直接当反例。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train]"
python -m training.wakeword all --config training/wakeword/configs/siu_maau_jan.yaml
python -m training.wakeword install --config training/wakeword/configs/siu_maau_jan.yaml   # 复制到包里本机试用
python scripts/wakeword_model.py publish v3 --notes "改了什么、真机表现" --push            # 发布成版本分支
```

### 网页版测试（浏览器麦克风）

```bash
catman-io webdemo --open          # 默认 http://127.0.0.1:8765 ，--record-dir 指定录音目录
```

页面里点"开始监听"，对着麦克风说「小貓人」，能看到实时分数、命中记录、折合每小时的命中次数；
"保存最近 3 秒"会把录音存成 16 kHz WAV，正好用来收集真人正样本和误唤醒片段。浏览器只允许
`http://localhost` 或 https 页面用麦克风，在别的机器上开页面要走 SSH 隧道（`ssh -L 8765:127.0.0.1:8765 <主机>`）。

## 配置

见 [config.example.yaml](config.example.yaml)，每个键都有注释。密钥一律不写进 YAML，用 `*_env` 指向环境变量。
要点：`audio.device` / `output_device`（设备编号或名字子串，`catman-io devices` 查）、`actions.weather` 的经纬度、
`intent.llm` 与 `brain.llm`（OpenAI 兼容端点，Gemini 的兼容端点也行）、`brain.catman`（delegate 用）、
`api.host`（给 Docker 里的 catman 访问要改成 `0.0.0.0`）。

## 开发

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train,audio,asr,tts,demo]"
catman-io setup
python scripts/wakeword_model.py pull                    # 没有模型时相关测试会跳过
ruff check catman_io training tests && pytest -q
CATMAN_IO_NETWORK_TESTS=1 pytest -q -m network            # 要联网的（edge-tts）
CATMAN_IO_TEST_ASR_ROOT=data/models/asr pytest -q tests/test_asr.py   # 有识别模型时
```

## 路线图

1. ✅ 唤醒词（粤语）、端点检测、粤语识别、三层意图、动作、对话模型、粤语合成、整条流水线
2. ✅ 回合日志与 bad case 飞轮、给 catman 的 HTTP API 与技能
3. 用真人录音重训唤醒词；在目标设备上校准阈值与端点参数
4. catman 侧 voice 渠道，让 delegate 的结果直接从扬声器出来（协议草案见 integrations/catman/README.md）
5. 网页 demo 扩成整条流水线；音频流推送给 live 模型；离线粤语 TTS；智能家居预置意图
