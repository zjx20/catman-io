---
name: catman-io
description: 维护粤语语音助手 catman-io 的意图规则：拉取 bad case 复盘、改现场规则、干跑回归、写回并热加载。用户说"优化语音意图 / 处理 bad case / 语音助手听唔明"或定时任务触发时用。
---

# catman-io 意图飞轮

catman-io 是跑在语音小主机上的粤语语音助手（唤醒词「小貓人」）。它把每一回合记进日志并自动标出
bad case；你的工作是定期把这些 bad case 变成更好的规则和回归用例。一切通过它的 HTTP API 做。

## 连接

- 地址：环境变量 `CATMAN_IO_API_URL`（例如 `http://host.docker.internal:8766`）
- 令牌：环境变量 `CATMAN_IO_API_TOKEN`，每个请求带头 `X-Catman-IO-Token: $CATMAN_IO_API_TOKEN`
- 先探活：`curl -s $CATMAN_IO_API_URL/api/health`

下面的命令都假设：
```bash
H="X-Catman-IO-Token: $CATMAN_IO_API_TOKEN"; U="$CATMAN_IO_API_URL"
```

## 流程

1. **拉复盘包**：`curl -s -H "$H" "$U/api/journal/review?format=md"`（JSON 版去掉 `format=md`）。
   里面按标记分组，每条有原话 `text`、命中的意图、回复、标记详情。标记含义：
   - `rules_miss_llm_hit`：规则没中、LLM 判出了已知意图 → **最有价值**：把这句的说法补进该意图的 `patterns`，
     并把复盘包末尾的候选用例追加为回归用例。
   - `rules_llm_disagree`：规则命中但 LLM 不同意 → 看规则是不是误命中，必要时收紧模式或加 `negatives`。
   - `fallthrough_chat` / `llm_chat`：没判出意图 → 看看是不是缺一个意图；不是指令的话不用管。
   - `user_retry` / `user_negation`：用户马上重说或否定 → 上一回合多半错了，看它命中了什么。
   - `wake_no_speech`：误唤醒，规则层不用管（唤醒词另外重训）。
   - `asr_empty` / `asr_short`：没听清，规则层不用管。
   - `slow`、`llm_timeout`、`brain_error`、`action_failed`：性能或后端问题，记下来告诉用户。
2. **看当前规则**：`curl -s -H "$H" $U/api/intent/rules`（文件与意图清单），
   `curl -s -H "$H" $U/api/intent/rules/site.yaml`（现场规则，可能还不存在 → 404，就从空文件开始）。
   `builtin.yaml` 只读，同名意图写在 `site.yaml` 里会覆盖它，所以要改内置意图就把整个意图复制到 site.yaml 再改。
3. **改规则**：规则文件格式见下。模式写在归一化后的文本上（无标点、小写、香港繁体；简体也会被转成繁体）。
   每个意图的 `examples` 必须全部命中本意图（lint 会查），所以把 bad case 的原话加进 `examples`。
4. **干跑**：`curl -s -H "$H" -X POST $U/api/intent/test -H 'Content-Type: application/json' \
     -d "$(jq -n --rawfile y site.yaml '{rules: {"site.yaml": $y}}')"`
   返回 `errors`（必须为空）、`warnings`、`cases.failures`。失败就继续改。
5. **写回**：`curl -s -H "$H" -X PUT $U/api/intent/rules/site.yaml --data-binary @site.yaml`
   通过才会写盘并立刻生效（旧文件留 `.bak`）；409 带同样的报告。
6. **追加回归用例**：`curl -s -H "$H" -X POST $U/api/intent/cases -H 'Content-Type: application/json' \
     -d '{"cases": [{"text": "聲音調去三成", "intent": "volume.set", "slots": {"percent": 30}, "source": "catman"}]}'`
   不该命中任何规则的句子写 `"intent": "none"`。用例会保护以后的改动。
7. **ack**：`curl -s -H "$H" -X POST $U/api/journal/review/ack -H 'Content-Type: application/json' -d '{"until": <复盘包里的 until>}'`
8. **汇报**：一两句话说改了哪些意图、加了几条用例、还有什么处理不了（性能、后端）。

## 规则文件格式

```yaml
version: 1
intents:
  - name: timer.set              # 意图名；和 builtin 同名即覆盖
    description: 設定倒數計時     # 也是给 LLM 层看的工具说明
    slots: {duration: duration}   # 槽位名: 类型；类型带 ? 表示可选，如 {date: "date?"}
    examples: ["幫我set個十分鐘嘅timer", "三個字之後叫我"]   # 必须全部命中本意图
    negatives: ["十分鐘前發生咗乜嘢"]                       # 不得命中本意图
    patterns:
      - "^{polite}(?:set|較|校)(?:一)?個?{duration}(?:嘅)?(?:timer|鬧鐘){tail}$"
    action: builtin.timer         # 内置动作名，或 http 动作（见下）
    priority: 0                   # 冲突时大的优先
  - name: light.on
    slots: {room: "room?"}
    examples: ["開燈", "幫我開廳燈"]
    patterns: ["^{polite}開(?:埋)?({room})?(?:盞)?燈{tail}$"]
    action: {type: http, method: POST, url: "http://homeassistant.local:8123/api/services/light/turn_on",
             headers: {Authorization: "Bearer ${HASS_TOKEN}"}, json: {entity_id: "light.{room|living_room}"}}
    say: "開咗{room|}燈喇"
```

- 槽位类型：`number`、`duration`（秒）、`time`、`date`、`percent`、`room`、`text`。
- 宏：`{tail}` = 句尾语气词（呀 / 啦 / 喎…），`{polite}` = 句首客气话（唔該 / 幫我…）。
- 内置动作：`builtin.time`、`builtin.date`、`builtin.timer`、`builtin.timer_cancel`、`builtin.volume_up`、
  `builtin.volume_down`、`builtin.volume_set`、`builtin.mute`、`builtin.stop`、`builtin.repeat`、`builtin.help`、
  `builtin.weather`。
- 一个意图新增时，LLM 层会自动拿到对应的工具，不用另外配置。

## 注意

- 不要为了让用例通过而删用例；用例失败说明改动影响了旧行为，要么修规则，要么确认旧用例本身错了再改它。
- 不要把闲聊句子硬做成规则；规则只管明确的指令。
- 改完用 `/api/intent/parse` 抽几句 bad case 验证：`-d '{"text": "聲音調去三成"}'`。
