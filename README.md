# botlab (Python SDK + CLI)

BotLab 的 Python SDK 与命令行工具：用 `project()` / `Robot` / `Model` / `Sensor` /
`Motion` / `Scene` 在 `controller.py` 里声明机器人工程，把控制逻辑写成信号表达式，
然后在本地一条命令把整个工程编译、打包并跑进浏览器版 BotLab 仿真运行时。

```
pip install botlab
cd my-robot-project
botlab run
```

`botlab run` 是纯调试循环，全程不写任何 zip 文件。它托起**完整的 BotLab 编辑器**
（不是发布页）：左侧文件树、Monaco 编辑器、Console / Watch / Metrics 面板和线
上 IDE 完全一致，只是工程来自本地目录而不是云端：

1. 导入 `controller.py`，收集工程声明（资源、参数、频率）；
2. 执行 `main.py`，把 `PolicyController` 的信号表达式降解成控制 plan；
3. 在内存里组装 release（STORED 不压缩、同格式完整性校验），托管给浏览器；
4. 起本地静态服务并自动打开浏览器，Vite 风格横幅打印可点击链接
   （默认绑定 `0.0.0.0`，`Local` + `Network` 两条，局域网设备可访问）：
   打开的 `/` 就是编辑器——本地快照的 `index.html` 带有
   `window.BOTLAB_LOCAL_PROJECT` 标记（URL 不带任何后缀），编辑器据此从
   `local-project.json` 实时读取本地目录装载工程；`/embed.html` 仍可手动
   打开预览发布形态。

启动输出保持 Vite 式的两三行（`microduck-project ready` + Local/Network 链接）；
编译摘要（资源清单、plan 规模、包大小）收在 `--verbose` 之后。启动后**改源码、
刷新页面**即可：编辑器重新读取本地目录，release 也会在子进程里重新编译，无需
重启命令。要导出可发布/可上传的 `release.zip`（DEFLATED 压缩、落盘），用
`botlab build`。

CLI 完全通用：任何符合 BotLab 两文件契约（`main.py` + `controller.py`）的工程都能
跑，不针对特定机器人或策略。

## 工程结构

```
my-robot-project/
├── main.py                 # 固定四行入口
├── controller.py           # 工程声明 + 控制表达式
├── robot.preset.yaml       # 关节顺序、PD、初始状态等资产标定
└── assets/                 # MJCF、ONNX、mesh 等项目资产（相对路径引用）
```

`main.py` 必须是标准四行（release 的 compiled controller 契约）：

```python
from botlab import Simulation
from controller import create_controller

sim = Simulation.from_project()
sim.run(create_controller(sim))
```

`controller.py` 按平台的通用组件写观测、模型与输出，与浏览器内编辑器里的写法
逐字一致——同一份代码在两边都跑得通：

```python
from botlab import Model, PolicyController, Robot, concat, project

project("My Robot", control_hz=50, physics_hz=200)
robot = Robot("assets/robot/scene.xml", preset="robot.preset.yaml")
actor = Model("assets/models/policy.onnx")

def create_controller(sim):
    observation = concat([robot.projected_gravity(), robot.joint_position(default=robot.default_angles)])
    action = actor(observation).action()
    target = robot.default_angles + robot.action_scale * action[robot.motor_order]
    return PolicyController(robot, control=target)
```

API 契约的完整文档见 BotLab 仓库的 `docs/api/`（Signal 表达式、`policy_bank` 多
策略、`state_machine` 状态机、preset 字段表等）。

## 命令

| 命令 | 作用 |
|------|------|
| `botlab run [DIR]` | 调试运行：源码内存编译 + 浏览器仿真，改源码后刷新页面即生效（不写 zip；`--verbose` 打印编译摘要；`--out` 可切换为落盘导出模式；`--no-browser`、`--host/--port`） |
| `botlab build [DIR]` | 本地导出 `release.zip`（默认写到工程目录；`--out` 自定义路径）。产物与平台 `/d/<id>/` demo 发布包同格式，可直接交由平台托管 |
| `botlab validate [DIR]` | 只做声明收集 + plan 编译 + 打包演练，不起服务 |
| `botlab ui sync --botlab <path>` | 从一个 BotLab 构建产物同步完整 UI 快照（编辑器 + embed 运行时） |
| `botlab ui status` | 查看当前内置 UI 快照状态 |

## 文档（给人和给 AI 的手册）

- [docs/component-reference.md](docs/component-reference.md) — **组件速查表**：SDK 全部公开符号与契约，写代码前先查
- [docs/decision-table.md](docs/decision-table.md) — **选型决策表**：需求 → 组件映射、代码分层判断法、常见错误模式
- [docs/error-guide.md](docs/error-guide.md) — **报错对照**：错误关键词 → 原因 → 修复
- 完整 API 语义与教程见 BotLab 仓库 `docs/api/`、`docs/examples/`

`botlab run` 默认绑定 `127.0.0.1` 的随机空闲端口；服务需要发送
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` 隔离头（MuJoCo /
ONNX Runtime Web 的 `SharedArrayBuffer` 依赖），代理或反代时必须保留这些头。
静态资源请求不打印访问日志（一次页面加载数百个 chunk 是噪声）；错误（4xx/5xx）
仍会打印，`log_request` 被禁用不影响 `send_error` 路径。

## UI 快照（`botlab/ui/`）

包内自带的浏览器运行时是 BotLab 构建产物（`dist/`）的完整镜像（去掉 `d/` demo
应用；`index.html` 里的统计脚本本地剥离），由 `botlab ui sync` 从 BotLab checkout
同步。快照里的 UI 与 botlab 平台前端同源——本地调试的编辑器、线上 IDE 和
`/d/<id>/` 发布页共用同一套运行时。快照目录被 gitignore（属于构建产物），打包时
通过 hatch 的 force-include 进入 wheel。

## 运行机制

浏览器端 embed 运行时读取 `./release.zip`：`release.json`（manifest、编译后的
controller plan、SHA-256 完整性清单）、`project/` 源码与 `assets/` 二进制。带
compiled controller 的 release 不需要浏览器端 Pyodide —— plan 由 TypeScript 仿真
引擎（MuJoCo worker + ONNX Runtime Web）逐帧执行，这正是 BotLab 的热路径。

限制（与平台契约一致）：

- `main.py` 必须是标准四行。
- `Signal.action(modify=...)` 回调：`botlab run` 支持运行（浏览器内 Pyodide 执行
  main.py，工程以「源码 + manifest」形态进包，无 compiledController）；`botlab build`
  导出的离线包同样可跑（任何 BotLab embed 运行时都能起 Pyodide），但发布形态比
  compiled 包重（多一次 Python 运行时加载，每控制步一次回调边界）。自定义
  `Controller.step()` 逐步回调同样走这条路径。
- 非空 `botlab.yaml` 的旧工程不在 CLI 支持范围内（Python 声明是唯一入口）。
- 工程资产按项目相对路径声明，必须真实存在于工程目录内。

## 开发

```
uv venv && uv pip install -e .[dev]
pytest
```

`src/botlab/sdk.py` 与 BotLab 仓库的 `src/domain/python/botlab.py` 保持同步
（同一份 SDK 双端运行）；`_host.py` 是浏览器 JS bridge 的 CPython 对应物。

## License

Proprietary — 与 BotLab 主仓一致。
