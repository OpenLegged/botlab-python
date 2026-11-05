# BotLab HIL 设备链路部署 runbook（AI 助手专用）

本文件是 BotLab 网页版「设备网关」面板提示词的**目标文档**：面板提示词只需一句话指向本文件。
读者是用户委托的 AI 助手。按本文件执行即可把「浏览器 ↔ 网关 ↔ 设备 agent」链路跑通。

**目标**：让浏览器「设备网关」面板连上网关，模型推理下发到目标设备执行。

## 0. 先问场景，再动手

用一句话向用户确认处于哪种情况，确认前只做只读检查：

- **A. 本机试跑（无板卡）**：同一台电脑起网关 + agent，浏览器连 `ws://127.0.0.1:50060`。
- **B. 真实 Linux 板卡（板卡 HIL 需 Python ≥3.11，系统 Python 更旧时传独立 CPython，见第 4 步），网线/局域网连着用户本机且 SSH 可达**：网关 + agent 都跑在板卡回环，本机一条 `ssh -L` 隧道把网关投到本机回环，浏览器仍连 `ws://127.0.0.1:50060`。板卡装包一律**离线装**（第 4 步），不要求板卡出网。异地无法 SSH 直达的板卡才需要域名服务器方案（见 `hil.md`「Gateway targets」）。

## 1. 组件检查（只读）

- `which botlab-hil-gateway botlab-agent botlab`；若存在，看脚本 shebang 定位所属 Python 环境（`~/.local/share/uv/tools/<name>/bin/python` = uv tool 安装），用对应方式查版本（`uv tool list` / `python3 -m pip show botlab-python`）。
- 若本机有该仓库 checkout，`git fetch origin feat/control-protocol-core` 后 `git rev-list --count HEAD..origin/feat/control-protocol-core`，>0 才需要更新。
- 用 `importlib.util.find_spec` 探测工具环境内 `websockets` / `numpy` / `onnxruntime` 是否齐备，缺哪个补哪个 extras。

## 2. 安装 / 更新

- 源：`git+https://github.com/OpenLegged/botlab-python.git@feat/control-protocol-core`
- ⚠️ **git 源构建必须前缀 `BOTLAB_ALLOW_MISSING_UI=1`**：UI 快照（`src/botlab/ui`）是生成产物不进 git，否则构建钩子拒绝出包。HIL 场景用不到 UI，此为官方支持用法。
- 本地 checkout 若含 `src/botlab/ui`，可直接从该目录安装（同一提交、免开关、保留 UI，严格更优）。
- extras：`[gateway]` = websockets（网关机）；`[target]` = numpy + onnxruntime + websockets（设备端推理必需，缺了 agent 能注册但跑不了模型）；`[runtime]` = 本地 MuJoCo 仿真，HIL 不需要。
- **板卡等无外网机器一律离线装**：助手机构建本包 wheel + 按目标机标签下载依赖 wheel，scp 后 `pip install --no-index --find-links`（完整命令见第 4 步）。不要靠临时共享网络或蹭 Wi-Fi 在线装。
- **HIL 硬门槛 Python ≥3.11**：`websockets>=17.1` 的 Requires-Python 是 ≥3.11，且拨号代码（`hil/wsstream.py`）传 17.1 才有的 `legacy=True`，降到 16.x 会直接 TypeError。板卡系统 Python 更旧时用独立 CPython（第 4 步）。离线装的 wheel 的 cp/ABI 标签必须与板卡 venv 实际 Python 一致（cp310 wheel 在 3.12 venv 里 pip 视而不见，报 `from versions: none`）。
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

**先做硬件适配**：SSH 可达后先识别板卡——`cat /proc/device-tree/model`、
`. /etc/os-release && echo "$PRETTY_NAME"`——再对照仓库 [`skills/`](../skills/README.md)
目录（读各 `skills/*/SKILL.md` 的 frontmatter）。命中则把该 skill 作为本节的板卡侧
补充（默认组网、系统坑、实测版本都写在里面，只加载命中的那一份）；未命中则按本节
通用命令走，跑通后欢迎按 `skills/README.md` 回来贡献一个。

```bash
# 助手机（有外网 + 本仓库 checkout）：构建本包 wheel + 按板卡标签下载依赖 wheel
# --python-version/--abi 必须等于板卡 venv 将使用的 Python 版本（示例 3.12/aarch64）
uv build --wheel   # 本地 checkout 含 src/botlab/ui，免 BOTLAB_ALLOW_MISSING_UI
python3 -m pip download -d botlab-pkgs --no-cache-dir --only-binary=:all: \
  --python-version 3.12 --abi cp312 --implementation cp \
  --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 \
  --platform manylinux_2_27_aarch64 --platform manylinux_2_28_aarch64 \
  numpy "onnxruntime>=1.18" "websockets>=17.1" "pyyaml>=6.0"
cp dist/botlab_python-*.whl botlab-pkgs/
scp -r botlab-pkgs <用户>@<板卡地址>:~/

# 板卡建 venv：系统 Python ≥3.11 时直接
python3 -m venv ~/botlab-venv
# 系统 Python <3.11（websockets>=17.1 要求 ≥3.11）或 venv 报缺 ensurepip 时，
# 先由助手机从 astral-sh/python-build-standalone releases 下载
# cpython-3.12.*-<板卡架构>-install_only.tar.gz（自带 ensurepip）scp 到板卡，然后：
mkdir -p ~/botlab-python && tar xzf cpython-*.tar.gz -C ~/botlab-python --strip-components=1
~/botlab-python/bin/python3 -m venv ~/botlab-venv

# 板卡：离线安装（全程不要求板卡出网）
~/botlab-venv/bin/pip install --no-index --find-links ~/botlab-pkgs "botlab-python[gateway,target]"

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
- 离线装报 `from versions: none` → wheel 的 cp/ABI 标签与板卡 venv 的 Python 不一致，按 venv 实际版本重新 `pip download`。
- 装不上 `websockets>=17.1`（17.x Requires-Python ≥3.11）→ 板卡 Python 太旧，用独立 CPython 建 venv；不要降级 16.x（拨号传 `legacy=True` 会 TypeError）。
- 浏览器连不上但板卡 agent 已注册 → 本机 `ssh -L` 隧道断了或本机 50060 被占，`lsof -nP -iTCP:50060` 查明后重跑隧道命令。
- 板卡特有报错（系统时钟漂移、远端 `pkill` 自匹配、系统 Python 缺组件等）→ 先查 `skills/<硬件>/SKILL.md` 的实测坑清单，再按通用路径排查。

架构与协议背景见 [hil.md](hil.md)。
