# BotLab Control CLI 部署 runbook（AI 助手专用）

本文件是 BotLab 网页版「CLI 控制」面板提示词的**目标文档**：面板提示词只需一句话指向本文件。
读者是用户委托的 AI 助手。按本文件执行即可把「CLI → 通用中间件 → 浏览器页面」链路跑通。

**目标**：让用户正在使用的 BotLab 网页（编辑器或托管页）连上本地通用中间件，
此后通过 `botlab control` 命令驱动页面里的机器人仿真。

## 0. 链路与前提

- 用户已有浏览器页面，浏览器侧**无需安装任何东西**；本地要安装的只有
  `botlab-python`（提供 `botlab` CLI 与通用中间件）。
- 中间件是**通用的**：`botlab control serve` 一条命令启动，不含任何项目代码。
  项目的能力声明（capabilities）、动作表与控制经验由**页面在配对时上报**，
  中间件按描述符构建——因此**不需要检出被控制项目的源码**。
- 中间件是 WS 服务端，浏览器作为客户端拨入，因此必须跑在用户本地（或用户可达的机器上）。
- 动手前向用户确认一件事：从哪个地址打开的 BotLab 页面（决定第 3 步的 `--origin`）。
- 项目自带中间件（如 microduck 的 `python -m microduck_control serve`）仍受支持：
  按项目自己的说明启动即可，后续步骤不变；本文件默认走通用中间件。

## 1. 组件检查（只读）

- `which botlab`；若存在，看脚本 shebang 定位所属 Python 环境，用对应方式查版本
  （`python3 -m pip show botlab-python`）。
- 版本号不能区分是否含 `control`；以 `botlab control --help` 是否存在为准。

## 2. 安装 / 更新

- 源：`git+https://github.com/OpenLegged/botlab-python.git@feat/control-protocol-core`
- ⚠️ **git 源构建必须前缀 `BOTLAB_ALLOW_MISSING_UI=1`**：UI 快照（`src/botlab/ui`）是
  生成产物不进 git，否则构建钩子拒绝出包。CLI 控制用不到 UI，此为官方支持用法。
- 本地 checkout 若含 `src/botlab/ui`，可直接从该目录安装（同一提交、免开关、保留 UI，严格更优）。
- 安装带 `serve` extra（中间件需要 aiohttp）：`pip install "botlab-python[serve]"`；
  控制客户端本身标准库 only。缺 extra 时 `botlab control serve` 报
  `missing_dependency`（exit 2），按提示补装即可。
- 系统 pip 受限先建 venv；Windows 用 `py -3 -m pip`。

## 3. 启动中间件并配对

```bash
botlab control serve --port 8776 --origin <本页 origin>

# 取配对信息（websocket 地址 + 令牌）：
botlab control connection --show-token --json
```

- 工作目录决定中间件的项目锚点（discovery 的 `project_key`）；与页面项目无关，
  在哪个目录启动，之后就在哪个目录（或加 `--project` 指向它）调用其余命令。
- 中间件默认只监听 127.0.0.1 且只接受 allowlist 的 origin。**网页版页面的 origin 不是
  127.0.0.1**，必须加 `--origin <页面地址的 scheme://host[:port]>`（如
  `--origin https://botlab.example.com`），否则浏览器的 WS 连接被拒。
- 每个中间件只接受一个浏览器运行实例；不要对同一浏览器同时启动两套服务。
- 把 `connection --show-token` 输出的 websocket 地址与令牌**在对话中**交给用户，
  提醒其粘贴到浏览器「CLI 控制」面板并点「连接并启用外部控制」。
  连接只启用外部指令，不会让机器人移动。

## 4. 验证

```bash
botlab control state --json
```

`connected` 与 `external_enabled` 均为 `true`、`run_state` 为 `running`、`ready` 为
`true` 时链路就绪。`ready=false` 常见于页面隐藏或仿真暂停——保留配对，恢复后即可发令。

## 5. 控制规则（接管后）

1. 控制前先拉项目经验与能力表（均由页面上报）：
   `botlab control experience --json` 与 `botlab control capabilities --json`；
   经验记录实测参数（如触发行走的最低速度）与坑，capabilities 声明不含这些。
2. 每个动作：读 `state` → 发命令 → `wait --id` 直到 `terminal:true` → 核对物理反馈，
   再发下一条。
3. 工具输出解析失败不等于命令未执行：先查 `state` 的 `active_command_id`，
   确认未接受后才可重发。
4. 参数按 `capabilities` 声明的范围发送；动作别名来自页面声明，通用 CLI 不内置；
   定时移动不是精确导航，实际效果以位置与姿态反馈为准。
5. 网络超时只表示结果未知：先 `wait` 原请求，必要时对同一 request-id 显式 `retry`；
   新的运动决定用新 ID。
6. 退出码：`0` 为请求成功（含 `pending`）、`2` 为请求/连接错误、`3` 为任务失败/取消/过期。

## 6. 安全规则

令牌只在用户对话中转述，绝不写入文件、日志或提交；令牌经 `BOTLAB_CONTROL_TOKEN`
或隐藏输入传入时不入 argv、不落 shell 历史。请求收据目录（默认
`~/.local/state/botlab/control/receipts/`）保持 0700/0600，不复制进项目日志，也不要
删除收据来绕过冲突。

## 7. 出错纪律

贴完整报错、分析根因后再行动，不盲目换端口或重装。已知坑速查：

- 浏览器面板连不上、中间件日志显示 origin 拒绝 → 缺 `--origin`，按第 3 步补页面 origin；
- `state` 显示 `connected:false` → 用户尚未在面板完成配对，重新走第 3 步交付流程；
- `capabilities` / `experience` / 发令返回 `no_project_paired` → 页面还没配对上报描述符，
  或页面过旧不支持下发；确认用户用的是新版页面并重新连接；
- `serve` 报 `missing_dependency` → 缺 serve extra，按第 2 步补装 aiohttp；
- 中间件重启后旧命令全部被拒 → 实例身份（instance_id）已更换，属预期；重新配对即可，
  不要重放旧 session / episode / request ID；
- CLI 安装报 `src/botlab/ui is missing` → git 源缺 UI 快照，安装命令加前缀
  `BOTLAB_ALLOW_MISSING_UI=1`；
- 端口相关报错 → 先 `lsof -nP -iTCP:8776` 查占用再决定，不要直接换端口。

协议与托管背景见 [control-protocol.md](control-protocol.md) 与 [control-hosting.md](control-hosting.md)。
