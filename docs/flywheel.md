# 意图飞轮：bad case 自动收集与改进

目标：不用人盯着，也能让规则越用越准。

## 回合日志

`catman-io run` 把每一回合写进 `data/journal/<日期>.jsonl`，一行一个回合：识别文本、命中的意图与层级
（rule / llm / none）、槽位、动作、回复、各段延迟（说完到开口是主指标），以及音频文件路径
（唤醒前后那段、整句）。

自动标记（详见 `catman_io/journal/flags.py`）：

| 标记 | 意思 | 谁来处理 |
|---|---|---|
| `rules_miss_llm_hit` | 规则没中、LLM 判出已知意图 | 规则缺覆盖 → 补模式 + 加用例（最有价值） |
| `rules_llm_disagree` | 规则命中但影子 LLM 不同意 | 看是否误命中 |
| `fallthrough_chat` / `llm_chat` | 没判出意图 | 看是否缺意图 |
| `user_retry` / `user_negation` | 用户马上重说 / 否定 | 上一回合多半错了 |
| `wake_no_speech` | 唤醒后没人说话 | 误唤醒样本 → 重训唤醒词 |
| `asr_empty` / `asr_short` | 没听清 | 看录音 / 换 ASR 模型 |
| `barge_in_early` | 回复 2 秒内被打断 | 回复不对或太长 |
| `slow` / `llm_timeout` / `brain_error` / `action_failed` | 性能或后端 | 看配置 / 网络 |

看日志：

```bash
catman-io journal list --bad --since 1d
catman-io journal show <turn_id>
catman-io journal review -o data/journal/review.md     # 按标记分组的复盘（Markdown）
```

## 回归用例

`data/intent/cases.jsonl` 一行一个：`{"text": "...", "intent": "timer.set", "slots": {"duration": 600}}`，
`"intent": "none"` 表示不该命中。`catman-io intent test` 跑一遍，改规则前后都该是绿的。
LLM 判出的意图会自动变成"候选用例"出现在复盘里，确认后追加即可。

## 让 catman 来改

`catman-io run` 内嵌一个 HTTP API（默认 `127.0.0.1:8766`，令牌在 `data/api_token`），catman 的助手按
`integrations/catman/skills/catman-io/SKILL.md` 操作：拉复盘 → 改 `site.yaml` → 干跑 → 写回（自动 lint +
跑用例，过了才生效并热加载）→ 追加用例 → ack。接法见 `integrations/catman/README.md`。

自己动手也一样：改 `data/intent/rules/site.yaml`，运行中的 catman-io 会自动重新加载（坏文件不生效）。
