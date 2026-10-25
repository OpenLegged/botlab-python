# BotLab HIL 设备链路部署 runbook（AI 助手专用）

本文件是 BotLab 网页版「设备网关」面板提示词的**目标文档**：面板提示词只需一句话指向本文件。
读者是用户委托的 AI 助手。按本文件执行即可把「浏览器 ↔ 网关 ↔ 设备 agent」链路跑通。

**目标**：让浏览器「设备网关」面板连上网关，模型推理下发到目标设备执行。

## 0. 先问场景，再动手

用一句话向用户确认处于哪种情况，确认前只做只读检查：

- **A. 本机试跑（无板卡）**：同一台电脑起网关 + agent，浏览器连 `ws://127.0.0.1:50060`。
- **B. 真实 Linux 板卡（Python 3.10+），网线/局域网连着用户本机且 SSH 可达**：网关 + agent 都跑在板卡回环，本机一条 `ssh -L` 隧道把网关投到本机回环，浏览器仍连 `ws://127.0.0.1:50060`。板卡装包需出网。异地无法 SSH 直达的板卡才需要域名服务器方案（见 `hil.md`「Gateway targets」）。

## 1. 组件检查（只读）

- `which botlab-hil-gateway botlab-agent botlab`；若存在，看脚本 shebang 定位所属 Python 环境（`~/.local/share/uv/tools/<name>/bin/python` = uv tool 安装），用对应方式查版本（`uv tool list` / `python3 -m pip show botlab-python`）。
- 若本机有该仓库 checkout，`git fetch origin feat/control-protocol-core` 后 `git rev-list --count HEAD..origin/feat/control-protocol-core`，>0 才需要更新。
- 用 `importlib.util.find_spec` 探测工具环境内 `websockets` / `numpy` / `onnxruntime` 是否齐备，缺哪个补哪个 extras。

## 2. 安装 / 更新

- 源：`git+https://github.com/OpenLegged/botlab-python.git@feat/control-protocol-core`
- ⚠️ **git 源构建必须前缀 `BOTLAB_ALLOW_MISSING_UI=1`**：UI 快照（`src/botlab/ui`）是生成产物不进 git，否则构建钩子拒绝出包。HIL 场景用不到 UI，此为官方支持用法。
- 本地 checkout 若含 `src/botlab/ui`，可直接从该目录安装（同一提交、免开关、保留 UI，严格更优）。
- extras：`[gateway]` = websockets（网关机）；`[target]` = numpy + onnxruntime + websockets（设备端推理必需，缺了 agent 能注册但跑不了模型）；`[runtime]` = 本地 MuJoCo 仿真，HIL 不需要。
- 系统 pip 受限先建 venv；Windows 用 `py -3 -m pip`；uv 工具化环境用 `uv tool install --reinstall "botlab-python[...] @ <源>"`。

## 3. 场景 A 命令（本机试跑）

```bash
# 启动前确认端口空闲：lsof -nP -iTCP:50060
# 终端 1（令牌来自 BOTLAB_HIL_GATEWAY_TOKEN，未设则随机生成并由 --print-token 打印）
botlab-hil-gateway --host 127.0.0.1 --port 50060 --print-token
# 终端 2（省略 BOTLAB_HIL_TOKEN 则 getpass 交互输入，不落 shell 历史）
BOTLAB_HIL_TOKEN=<令牌> botlab-agent connect --gateway ws://127.0.0.1:50060 --device my-laptop
```

两个进程保持前台运行。验证标准：agent 日志出现 `BotLab HIL agent: registered as <id> (onnxruntime)`，且 `lsof -iTCP:50060` 显示 ESTABLISHED。助手可代跑（会话后台任务），但必须告知用户进程随会话结束而停止，并附自行重启的两条终端命令。

## 4. 场景 B 命令（板卡直连本机 + SSH 隧道）

两段 `ws://` 都不离开各自主机回环（浏览器拨本机回环、agent 拨板卡回环），跨网线
流量由 SSH 加密，因此不需要域名、证书或 `wss://`。

```bash
# 板卡（SSH 登录后执行；装包需出网——可临时把本机网络共享给网口，或让板卡连 Wi-Fi）
python3 -m venv ~/botlab-venv
BOTLAB_ALLOW_MISSING_UI=1 ~/botlab-venv/bin/pip install \
  "botlab-python[gateway,target] @ git+https://github.com/OpenLegged/botlab-python.git@feat/control-protocol-core"

# 板卡终端 1：网关（只绑板卡回环；未设 BOTLAB_HIL_GATEWAY_TOKEN 时随机生成并由 --print-token 打印）
~/botlab-venv/bin/botlab-hil-gateway --host 127.0.0.1 --port 50060 --print-token

# 板卡终端 2：agent（网关就在板卡上，拨板卡回环）
BOTLAB_HIL_TOKEN=<令牌> ~/botlab-venv/bin/botlab-agent connect \
  --gateway ws://127.0.0.1:50060 --device <设备ID>

# 本机终端 3：SSH 隧道（保持运行；断开后重跑同一条命令即可恢复）
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 50060:127.0.0.1:50060 <用户>@<板卡地址>
```

验证标准：板卡终端 2 出现 `BotLab HIL agent: registered as <id> (onnxruntime)`；
本机 `lsof -nP -iTCP:50060` 显示 ssh 监听 `127.0.0.1:50060`。

## 5. 安全规则

令牌只在用户对话中转述，绝不写入文件、日志或提交；`ws://` 仅限各自主机回环（场景 A 的本机回环、场景 B 的板卡回环），场景 B 的跨机流量只走 SSH 隧道；确需异地 `wss://` 网关时，强令牌一律 `secrets.token_urlsafe(32)`。

## 6. 交付

跑通后把「**网关端点 / 设备 ID / 令牌**」三项列表交给用户，并提醒：回浏览器「设备网关」面板粘贴三项后点「连接」。

## 7. 出错纪律

贴完整报错、分析根因后再行动，不盲目换端口或重装。已知坑速查：

- `src/botlab/ui is missing` → git 源缺 UI 快照，安装命令加前缀 `BOTLAB_ALLOW_MISSING_UI=1`；
- 端口相关报错 → 先 `lsof -iTCP:<port>` 查占用再决定，不要直接换端口；
- agent 能注册但推理报错 → 缺 `[target]` extras（numpy/onnxruntime），补装即可。
- 浏览器连不上但板卡 agent 已注册 → 本机 `ssh -L` 隧道断了或本机 50060 被占，`lsof -nP -iTCP:50060` 查明后重跑隧道命令。

架构与协议背景见 [hil.md](hil.md)。
