# 项目搭建指南（botlab-python / CLI 视角）

一个 BotLab 项目 = 两个固定入口 + `config/` 声明文件 + `assets/` 资产。浏览器编辑器与
本 CLI 消费同一份契约；本页讲本地从零搭一个可 `botlab run` 的项目。API 语义详见
BotLab 仓库的 `docs/api/`；组件速查见
[component-reference.md](./component-reference.md)。

## 标准骨架

```text
my-robot-project/
├── main.py                    # 固定四行入口，不改动
├── controller.py              # 声明段 + create_controller（observation/policy/output 三段）
├── config/                    # 声明数据（受限 YAML 子集），read_config("config/<name>.yaml") 引用
│   ├── runtime.yaml           # name、control_hz、physics_hz
│   ├── parameters.yaml        # 可调参数：label / default / range / step / unit
│   ├── keyboard.yaml          # 键盘绑定；gamepad.yaml 可选
│   ├── watch.yaml             # Watch 面板曲线
│   ├── metrics.yaml           # Metrics 面板读数
│   ├── control.yaml           # 外部控制面 control_profile（接 botlab control 才需要）
│   └── <设备名>.preset.yaml   # 按资源标定：robot、depth-camera、scene……
├── assets/
│   ├── robot/<机器人名>/*.xml + meshes/   # MJCF 与网格
│   ├── models/*.onnx                       # 策略网络
│   └── scenes/*.xml                        # 场景
└── <辅助模块>.py              # 复杂策略拆分；controller.py 显式 import
```

规则：

- preset 住 `config/`（`config/<name>.preset.yaml`）；根级 `*.preset.yaml` 是旧工程
  形态，读取兼容但不新建。
- 长向量（关节顺序、PD、初始姿态、相机外参）只进 preset，用通用 camelCase 字段名
  （`policyJointOrder`、`motorJointOrder`、`actionScale`、`defaultJointPositions`、
  `kp`、`kd`、`effortLimit`、`initialQpos`），不复制进 Python。
- 取决于策略训练方式的量必须显式声明：`history(..., layout=)` 恒显式（`"by-frame"` =
  IsaacGym、`"by-term"` = IsaacLab），观测顺序即列表顺序，减参考位姿是算法选择。
- 资源构造器用**字面量**路径；循环或 f-string 会把资源从工作区树里藏掉。
- 算法拓扑不进 preset：observation 顺序、history、归一化写在项目代码里。

### 目录规则

| 位置 | 规则 |
| --- | --- |
| 根级 | 只放 `main.py`、`controller.py` 与辅助 `*.py` 模块（显式 import，顶层段不得叫 `botlab`）；说明文档可放根级 |
| `config/` | 只接受一级 `config/<name>.yaml` 与 `config/<name>.preset.yaml`——无子目录、无点文件 |
| `assets/robot/` | **每台机器人一个以机器人名命名的文件夹**（含 `meshes/`）：相对引用按 XML 所在目录解析，命名文件夹隔离各机器人的 mesh 集；按类型分目录（`xml/`、`meshes/` 混放）是旧形态 |
| `assets/models/` | 单文件 ONNX 平铺；一组模型带附属文件时按模型组一个文件夹 |
| 其他资产 | 场景 XML 连同贴图放同目录；`motions/`、`meshes/`、`data/` 沿用惯例目录 |
| `import_tools/` | 仅动作导入类项目放 `file_rules.json` 等处理器声明 |
| 不要建 | `manifest.json`（平台模板专属）、根级 `*.preset.yaml`、`botlab.yaml`（旧形态） |

## 搭建步骤

1. **建骨架**：手工创建上面的目录；`main.py` 固定为：

   ```python
   from botlab import Simulation
   from controller import create_controller

   sim = Simulation.from_project()
   sim.run(create_controller(sim))
   ```

2. **放资产**：MJCF（连同 `meshes/`）、ONNX、场景 XML 放进 `assets/`。机器人 XML 的
   相对 include、mesh、纹理按其所在目录解析；所有资源必须在项目目录内（自包含）。

3. **写 preset**：`config/robot.preset.yaml` 以 `kind: "robot"` 开头，填关节布局、PD、
   初始状态；深度相机/场景同理（`kind: "sensor"` / `"scene"`）。

4. **controller.py 声明段**：`read_config` + `project(...)` + 资源构造器
   （`Robot` / `Sensor` / `Model` / `Motion` / `Scene`，字面量路径 + `preset=`）。

5. **create_controller 三段式**：observation（`robot.*` 观测 + `history`）→ policy
   （`Model(...)` 调用 + `.clip(...).action()`）→ output（`PolicyController`）。
   `target=` 走位置 PD；`control=` 直接写原生 MuJoCo 执行器控制，二者恰选其一。

6. **抽离声明数据**：超过几个键的声明进 `config/*.yaml`，源码只留结构与连线。
   命令轴在 `robot.commands(...)` 调用点声明一次；`keyboard.yaml` 只放绑定表。
   各类文件的最小形状：

   ```yaml
   # config/runtime.yaml
   name: My Robot
   control_hz: 50
   physics_hz: 200
   ```

   ```yaml
   # config/parameters.yaml —— 每项 label/default 必填，交互轴必须带 range
   target_velocity:
     label: {en: Forward speed, zh: 前进速度}
     default: 0.4
     range: [0, 0.5]
     step: 0.05
     unit: m/s
   ```

   ```yaml
   # config/keyboard.yaml —— 轴绑定 / 离散映射 / 复活键，按需取用
   velocity_command:
     bindings:
       target_velocity: {increase: KeyW, decrease: KeyS}
   respawn:
     key: KeyR
   ```

   ```yaml
   # config/gamepad.yaml —— stick: L|R 与 axis: 0|1；手柄复活键是 button:
   velocity_command:
     bindings:
       target_velocity: {stick: L, axis: 1}
     deadzone: 0.06
   ```

   ```yaml
   # config/watch.yaml / config/metrics.yaml —— 面板展示，经 project(panels={...}) 接入
   watch: [metric:forward_velocity]
   ```

   ```yaml
   # config/control.yaml —— 事件值必须与 commands 映射节点里的编码一致
   robot: MicroDuck
   state_node: state-machine
   vx: forward_velocity
   actions:
     - {name: sit, event_value: 1, allowed_states: [0], target_states: [2]}
   ```

   完整字段见 BotLab 仓库 `docs/api/`（parameters / controller / outputs / control 各篇）。

7. **复杂算法拆模块**：同目录 `.py`（顶层段不得叫 `botlab`），显式 import；辅助模块与
   显式表达式必须生成同一份 plan。要外部控制再加
   `declare_control(control_profile(**read_config("config/control.yaml")))`。

## 接线总览

项目里没有连线文件——每条线都是 `controller.py` 里的一次显式引用：

| 线 | 从 | 接到 | 在哪里接 |
| --- | --- | --- | --- |
| 资源标定 | `config/<设备>.preset.yaml` | 资源句柄 | 构造器字面量：`Robot(..., preset="config/robot.preset.yaml")` |
| 指令轴 | `parameters.yaml` 各参数的 `range` | `robot.commands(...)` 对应轴 | 调用点按参数名写轴；常量维写字面量 `0` |
| 设备输入 | `keyboard.yaml` / `gamepad.yaml` 绑定表 | 指令轴 | `robot.commands(..., keyboard=read_config(...)["<节>"], gamepad=...)` |
| 复活键 | `keyboard.yaml` / `gamepad.yaml` 的 `respawn` 键 | 项目声明 | `project(respawn={"keyboard": ..., "gamepad": ...})` |
| 面板 | 控制器 `metrics={...}` 命名 Signal | Watch / Metrics 面板 | `project(panels={...})`；yaml 用 `metric:<名字>` 引用 |
| 控制面 | `control.yaml` | `botlab control` / AI 控制 | `declare_control(control_profile(**read_config(...)))` |

三条红线：**字面量、无中间层**（preset 经构造器 `preset=` 字面量接线，不建装配
yaml）；**设备文件是纯绑定表**（只说哪个键驱动哪一维，互不镜像，向量只在 `commands`
调用点声明一次）；**名字即接线**（参数名、`metric:` 名、`event_value` 编码写错都在
编译期报错，无静默回退）。

## 命名与 label 体系

| 名字 | 写在哪 | 作用 | 形态 |
| --- | --- | --- | --- |
| `name=` | 资源构造器 | 资源 ID；同源多资源必须显式区分 | 字符串 |
| `_label` | `config/*.yaml` 顶层保留键 | 工程树叶名（展示名）；`read_config` 返回前剥离，不进声明数据 | 字符串或 `{en, zh}`（`en` 必填），非法形态硬失败 |
| `label`（preset 内） | `*.preset.yaml` | 资源语义字段，随 config 消费——与 `_label` 是两个键 | 字符串 |
| `label`（面板项） | `parameters.yaml` 每项、`metrics.yaml`、keyboard `mappings` | 显示文字，跟随界面语言 | 字符串或 `{en, zh}`（`en` 必填）；纯字符串所有语言原样显示 |

规范：面向用户的显示名一律写 `{en, zh}` 译文表并至少给 `en`；只有不需要本地化的
内部标识（preset `label`）用纯字符串。译文表 key 只能是 `en`、`zh`，缺译文回落 `en`。

## 验证环

```bash
botlab validate            # 收集声明、编译 plan、内存打包，不起服务
botlab run                 # 浏览器调试环：本地目录实时装载，刷新生效
botlab run --native        # 原生 CPython 执行（mujoco + onnxruntime），浏览器只读显示
botlab run --headless      # 无头跑（CI/冒烟），BOTLAB_STEPS=<n> 限定步数
botlab build               # 导出 release.zip（与浏览器发布同格式）
```

编译失败会点名缺失文件与字段；修完再验。动作导入类资产（NPZ/CSV）先在浏览器编辑器
走 `prepare_asset` 转换，再回到 CLI 迭代。

## 参考

- BotLab 仓库 `docs/concepts/project-setup.md` — 浏览器侧同一搭建路径与模板对照表
- [component-reference.md](./component-reference.md) — `botlab` 包全部公开符号
- [decision-table.md](./decision-table.md) — 需求到组件的映射与常见错误
- [onnx-export.md](./onnx-export.md) — 把策略网络导出为单文件 ONNX
