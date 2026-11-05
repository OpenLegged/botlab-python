# HIL 硬件适配 skills

每个子目录是一个 **HIL 硬件 skill**：给 AI 助手的*板卡特有*部署知识。部署 runbook
[`docs/hil-assistant.md`](../docs/hil-assistant.md) 在识别出板卡硬件后按需加载匹配的
skill（渐进加载：助手只读命中的那一份）。

**分层纪律**：通用流程（协议、令牌、隧道、离线装骨架、验证标准）都在 runbook，
skill 只写差异。两处重复的内容会在某处悄悄过期——写之前先问「这是这块板特有的吗」。

| skill | 硬件 | 实测版本 |
|---|---|---|
| [hil-rdk-x5](hil-rdk-x5/SKILL.md) | D-Robotics RDK X5（Sunrise OS） | `feat/control-protocol-core@d9b3bc1` |

## 贡献一个新硬件 skill

目录结构：

```
skills/hil-<硬件>/SKILL.md      # 目录名 = frontmatter 的 name
```

`SKILL.md` 要求：

- frontmatter 必填 `name`（`hil-<小写连字符>`，与目录同名）和 `description`
  （一句话中文，写清硬件 + 最关键的差异点）；`metadata` 写 `hardware` / `os` /
  `tested`（哪个 botlab 版本、何时实测）。
- 正文按固定骨架：
  1. **识别**——一条助手能独立判断「是不是这块板」的命令；
  2. **组网**——出厂默认地址 / 账户等；因用户环境而异的部分明确标「示例」；
  3. **与通用流程的差异 / 实测坑**——只写这块板特有且实测过的；
  4. **实测环境**——硬件、OS、botlab 版本、部署布局。

纪律（PR 审核按此口径）：

- **只写实测过的内容**，标注版本与日期；推测必须标「未验证」。
- 不把用户特定值写成假设（IP、路径等一律标示例）；**不含任何令牌 / 密钥**
  （出厂公开默认值除外，且注明「建议修改」）。
- 板卡侧**离线优先**；不提供联网一键脚本（`curl | sh` 之类）。
- skill 是会被 AI 助手直接执行的指令，视同代码：走 PR 审查进仓，不接收
  仓外地址动态加载。
