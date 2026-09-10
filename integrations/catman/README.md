# 接到 catman

catman-io 和 catman 之间现在只有两条线，都很轻：

1. **catman 改 catman-io 的意图规则**（飞轮）：catman-io 开一个本机 HTTP API，catman 的助手按
   `skills/catman-io/SKILL.md` 里的步骤拉 bad case、改规则、跑回归、写回。
2. **catman-io 把任务交给 catman**（`delegate`）：用户说"叫 catman 幫我……"时，catman-io 用管理员令牌
   `POST /api/chat` 发给 catman，回复照旧走微信。

一般对话（闲聊、问答）由 catman-io 自己的 LLM（`brain.llm`）处理，不经过 catman。

## 1. 让 catman 能访问 catman-io 的 API

catman 跑在 Docker 里，容器里能用 `host.docker.internal` 访问宿主机（compose 里已配 `host-gateway`）。
如果 catman-io 在另一台机器，就用那台机器的局域网地址。

`config.yaml`（catman-io 这边）：

```yaml
api:
  enabled: true
  host: 0.0.0.0      # 默认 127.0.0.1 只有本机能访问；给 Docker / 局域网访问要改成 0.0.0.0
  port: 8766
```

第一次 `catman-io run` 会在 `data/api_token` 生成令牌（也可以用环境变量 `CATMAN_IO_API_TOKEN` 指定）。

catman 这边，把这两个环境变量加进它的 `docker-compose.yml`（`catman` 服务的 `environment`），
助手的每个回合都会继承：

```yaml
      CATMAN_IO_API_URL: http://host.docker.internal:8766
      CATMAN_IO_API_TOKEN: <data/api_token 里那串>
```

## 2. 装技能

把 `skills/catman-io/SKILL.md` 复制到 catman 的技能目录（catman README「自助设置」一节说的
`$CLAUDE_CONFIG_DIR/skills/`，容器里通常是 `/data/claude/skills/`）：

```bash
mkdir -p ./data/claude/skills/catman-io
cp integrations/catman/skills/catman-io/SKILL.md ./data/claude/skills/catman-io/
```

下一回合起，微信里说"處理下語音助手嘅 bad case"，助手就会照着做。

## 3. 定时跑（可选）

用 catman 的 cron **agent 任务**（不是 shell 任务），每天跑一次。在 catman 的 dashboard 或用它的
`/api/me/cron` 建：

```json
{
  "name": "catman-io 意图飞轮",
  "schedule": "0 4 * * *",
  "task": {"kind": "agent", "prompt": "按 catman-io 技能处理最新的语音意图 bad case，改完汇报。", "session": "chain", "maxTurns": 40},
  "timeoutMs": 900000,
  "notify": {"end": true, "onlyFailure": false}
}
```

也可以由 catman-io 主动催：`catman-io flywheel nudge` 会往 catman 的管理员聊天发一句
"請按 catman-io 技能處理最新嘅 bad case"（要先配好 `brain.catman`）。

## 4. delegate：把任务交给 catman

`config.yaml`：

```yaml
brain:
  catman:
    base_url: http://192.168.1.1:8787   # catman dashboard 地址
    token_env: CATMAN_ADMIN_TOKEN        # catman 的管理员令牌放在这个环境变量里
```

需要 `intent.llm` 开着（由 LLM 判断哪些话是"要动手做的任务"）。发出去的是内置管理员身份，
所以和网页聊天页共用一个会话。

## 5. 以后：catman 侧的 voice 渠道（草案）

要让 catman 的回复直接从扬声器出来，得在 catman 加一个 `voice` 渠道（照 `src/channels/stdin.ts` 写，
约一百行），协议建议用一条 WebSocket、catman-io 作为客户端连上去：

```
catman-io → catman   {"type": "message", "msgId": "...", "userKey": "voice:<设备>:<人>", "text": "..."}
catman → catman-io   {"type": "send", "kind": "ack|progress|body", "text": "...", "messageId": "..."}
                     {"type": "typing", "on": true}
                     {"type": "recall", "messageId": "..."}
```

`kind` 让 catman-io 知道哪些要念（`body`）、哪些只是回执 / 进度。到时在 catman-io 加一个实现
`Brain` 接口的 `CatmanBrain` 即可，其他地方不用动。
