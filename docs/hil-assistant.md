# BotLab HIL 设备链路部署 runbook（AI 助手专用）

本文件是 BotLab 网页版「设备网关」面板提示词的**目标文档**：面板提示词只需一句话指向本文件。
读者是用户委托的 AI 助手。按本文件执行即可把「浏览器 ↔ 网关 ↔ 设备 agent」链路跑通。

**目标**：让浏览器「设备网关」面板连上网关，模型推理下发到目标设备执行。

## 0. 先问场景，再动手

用一句话向用户确认处于哪种情况，确认前只做只读检查：

- **A. 本机试跑（无板卡）**：同一台电脑起网关 + agent，浏览器连 `ws://127.0.0.1:50060`。
- **B. 真实 Linux 板卡（Python 3.10+，仅需出网）+ 有域名的服务器**：服务器跑网关 + TLS 反代，板卡跑 agent。

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

## 4. 场景 B 命令（板卡 + 域名服务器）

```bash
# 服务器（生成强令牌：python3 -c "import secrets; print(secrets.token_urlsafe(32))"）
python3 -m venv ~/botlab-venv
BOTLAB_ALLOW_MISSING_UI=1 ~/botlab-venv/bin/pip install \
  "botlab-python[gateway] @ git+https://github.com/OpenLegged/botlab-python.git@feat/control-protocol-core"
BOTLAB_HIL_GATEWAY_TOKEN=<强令牌> ~/botlab-venv/bin/botlab-hil-gateway \
  --host 127.0.0.1 --port 50060 --print-token        # 可改 read -s 输入避免进 shell 历史
caddy reverse-proxy --from <域名> --to 127.0.0.1:50060   # 需 DNS 指向服务器、80/443 可入站

# 板卡
python3 -m venv ~/botlab-venv
BOTLAB_ALLOW_MISSING_UI=1 ~/botlab-venv/bin/pip install \
  "botlab-python[target] @ git+https://github.com/OpenLegged/botlab-python.git@feat/control-protocol-core"
BOTLAB_HIL_TOKEN=<令牌> ~/botlab-venv/bin/botlab-agent connect \
  --gateway wss://<域名> --device <设备ID>
```

## 5. 安全规则

令牌只在用户对话中转述，绝不写入文件、日志或提交；`ws://` 仅限本机回环，跨机器一律 `wss://`；强令牌一律 `secrets.token_urlsafe(32)`。

## 6. 交付

跑通后把「**网关端点 / 设备 ID / 令牌**」三项列表交给用户，并提醒：回浏览器「设备网关」面板粘贴三项后点「连接」。

## 7. 出错纪律

贴完整报错、分析根因后再行动，不盲目换端口或重装。已知坑速查：

- `src/botlab/ui is missing` → git 源缺 UI 快照，安装命令加前缀 `BOTLAB_ALLOW_MISSING_UI=1`；
- 端口相关报错 → 先 `lsof -iTCP:<port>` 查占用再决定，不要直接换端口；
- agent 能注册但推理报错 → 缺 `[target]` extras（numpy/onnxruntime），补装即可。

架构与协议背景见 [hil.md](hil.md)。
