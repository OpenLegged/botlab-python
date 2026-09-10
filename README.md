# botlab (Python SDK + CLI)

BotLab 的 Python SDK 与命令行工具：用 `project()` / `Robot` / `Model` / `Sensor` /
`Motion` / `Scene` 在 `controller.py` 里声明机器人工程，把控制逻辑写成信号表达式，
然后在本地一条命令把整个工程编译、打包并跑进浏览器版 BotLab 仿真运行时。

```
pip install botlab
cd my-robot-project
botlab run
```

`botlab run` 会：

1. 导入 `controller.py`，收集工程声明（资源、参数、频率）；
2. 执行 `main.py`，把 `PolicyController` 的信号表达式降解成控制 plan；
3. 打包成 BotLab release.zip（v3 格式，含完整性校验）；
4. 起本地静态服务，托起 BotLab embed 运行时 + release 包，并自动打开浏览器。

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
| `botlab run [DIR]` | 编译打包并在浏览器里运行工程（`--no-browser`、`--out release.zip`、`--host/--port`） |
| `botlab validate [DIR]` | 只做声明收集 + plan 编译 + 打包演练，不起服务 |
| `botlab ui sync --botlab <path>` | 从一个 BotLab 构建产物同步 embed UI 快照 |
| `botlab ui status` | 查看当前内置 UI 快照状态 |

`botlab run` 默认绑定 `127.0.0.1` 的随机空闲端口；服务需要发送
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` 隔离头（MuJoCo /
ONNX Runtime Web 的 `SharedArrayBuffer` 依赖），代理或反代时必须保留这些头。

## UI 快照（`botlab/ui/`）

包内自带的浏览器运行时来自 BotLab 的发布产物闭包（`embed.html` + `assets/` +
`pyodide/` + `wasm/`，以 `dist/publish-assets.json` 为准），由 `botlab ui sync`
从 BotLab 构建同步。快照里的 UI 就是 botlab 平台的前端——本地运行与平台发布的
`/d/<id>/` demo 页共用同一运行时。

## 运行机制

浏览器端 embed 运行时读取 `./release.zip`：`release.json`（manifest、编译后的
controller plan、SHA-256 完整性清单）、`project/` 源码与 `assets/` 二进制。带
compiled controller 的 release 不需要浏览器端 Pyodide —— plan 由 TypeScript 仿真
引擎（MuJoCo worker + ONNX Runtime Web）逐帧执行，这正是 BotLab 的热路径。

限制（与平台契约一致）：

- `main.py` 必须是标准四行；`Signal.action(modify=...)` 与自定义 `Controller`
  逐步回调需要活的 Python 运行时，不能打进离线 release。
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
