# 组件速查表（AI 编码词汇表）

`controller.py` 能用的**全部**公开符号都在 `botlab` 包里，一个不多一个不少。
写代码前先查这张表：不在此表的 API 不存在，编造即报错。

## 资源构造器（模块顶层调用，声明进 manifest）

| 构造器 | 关键参数 | 说明 |
|--------|---------|------|
| `Robot(source, *, preset=None, config=None, name=None)` | `source`: MJCF 相对路径 | preset 载入关节布局/PD/初始状态 |
| `Model(source, *, provider=None, config=None, name=None)` | `provider`: auto/webgpu/webnn/wasm | ONNX 策略；同源多模型用 `name=` 区分 |
| `Sensor(robot, *, preset=None, config=None, name=None)` | — | 深度相机等传感器，挂在 robot 上 |
| `Motion(source, *, config=None, name=None)` | — | motion 参考数据 |
| `Scene(source, *, preset=None, config=None, name=None)` | — | 场景（含程序化 heightfield） |
| `project(name, *, control_hz=50, physics_hz=200, parameters=None)` | `parameters` 见下 | 工程声明；缺省频率 50/200 |

`project(parameters=...)` 每项：`{"label", "default", "range": [min, max], "step", "unit"}`，
`default` 必填且有限，`range` 内必须含 `default`，`step > 0`。

## 观测信号（从 robot/sensor 取，维度自动）

| 调用 | 维度 | 备注 |
|------|------|------|
| `robot.base_angular_velocity(scale=, source=)` | 3 | 自由根关节读取机体系角速度，无需再次逆旋转 |
| `robot.base_linear_velocity(scale=, source=)` | 3 | |
| `robot.projected_gravity(target=)` | 3 | `target="root"` 或 body 名 |
| `robot.base_height()` | 1 | |
| `robot.joint_position(order=, default=, scale=)` | actionDim | `default` 不自动填——减参考位姿是算法选择 |
| `robot.joint_velocity(order=, default=, scale=)` | actionDim | |
| `robot.last_action()` | actionDim | 上一帧 `Signal.action()` 的值 |
| `robot.imu(order=, scale=)` | — | |
| `robot.qpos(indices, default=, scale=)` / `qvel(...)` | len(indices) | 直接按 MuJoCo 索引读 |
| `robot.mujoco(query, size, target=, start=, scale=)` | size | 逃生口：`sensor.data` 等原生查询，`size` 必填 |
| `robot.commands(*names, ...)` | len(names) | 轴上下界来自参数 `range` |
| `sensor.depth(max_age=)` | 相机宽×高 | 超龄报错，无兜底 |
| `sensor.depth_frame(max_age=)` | data + age_steps | 共享一次采样 |

## 处理/组合节点（Signal → Signal）

| 调用 | 用途 | 关键契约 |
|------|------|---------|
| `concat(signals)` | 拼接 | 单项不加节点；`concat` 顺序即观测顺序 |
| `take(sig, indices)` / `sig[indices]` | gather | 允许短/重复索引；**单下标和切片报错** |
| `split(sig, sizes 或 {"name": size})` | 拆分 | 返回 tuple 或具名字典 |
| `constant(value, size=n)` | 常量 | 标量广播 |
| `sig.clip(low, high)` | 限幅 | 标量或等长向量 |
| `scale * sig + offset` | 仿射 | 折叠为单节点 |
| `history(src, size, stride=1, layout=, order=)` | 时间堆叠 | `layout="by-frame"`(IsaacGym) / `"by-term"`(IsaacLab) **必须显式** |
| `delay(src, total_steps, age=, fill=0.0)` | 显式总延迟 | `age` 接 `depth_frame().age_steps` 计入已发生延迟 |
| `where(cond, a, b)` | 逐帧二选一 | 两边都执行（不短路）；昂贵分支用 `mode_select` |
| `all_of / any_of / logical_not` 或 `& \| ~` | 布尔组合 | 操作数必须是布尔 Signal |
| `sig.gt/gte/lt/lte/eq/ne(other)` 或比较运算符 | 比较节点 | |
| `sig.at(i)` | 取标量 | 索引必须在维度内 |
| `resize(src, input_size=(w,h), output_size=(w,h), layout=, flip_x=, flip_y=)` | 图像缩放 | 尺寸顺序恒为 (width, height) |
| `rule_index(src, rules, default=0)` | 规则分类 | 首个匹配的整数值；rules 用比较节点构造 |
| `one_hot(src, size)` / `one_hot_index(src, threshold=, default=)` | 独热 | |
| `motion_reference(...)` | motion 参考信号 | 见 Robot/Controller API 文档 |

## 控制流（每帧分支的唯一合法形态）

| 调用 | 用途 | 备注 |
|------|------|------|
| `state_machine(command, initial=, transitions=[])` | 状态机 | 返回 `.state` 和 `.steps` 两个 Signal |
| `mode_select(mode, hold, choices={}, passive=0)` | 按模式选信号 | 全部分支同宽；元组键=状态共用；`passive=-1` 输出全零 |
| `mode_switch(command, initial=, transitions=[])` | 简单模式切换 | 无步数/条件触发时的简化形态 |
| `phase_clock(steps, period_steps=)` | 周期 cos/sin 相位 | 配合周期性技能 |

transition 行字段：`from`（默认 any）/ `command`（指令脉冲，优先级最高）/ `after_steps` /
`when`（布尔 Signal）/ `to`（必填）/ `reset_steps`（默认 True）。同行多条件**同时满足**才触发；
同级按声明顺序取首个匹配。

## 策略调用与输出

| 调用 | 用途 | 关键契约 |
|------|------|---------|
| `model(*inputs, input_name=, output_name=, constants=, control_decimation=, call_id=)` | 单模型推理 | 输出宽度来自 ONNX session；同模型多调用共享 session |
| `policy_bank(mode, inputs= 或 sources=, policies=[], control_decimation=, call_id=)` | 多策略调度 | 每 mode 唯一；`sources`+`fields` 是高层写法 |
| `sig.action(*, modify=None)` | 标记动作 | **全计划有且仅有一个**；`modify` 见限制章节 |
| `PolicyController(robot, *, target= 或 control=, kp=, kd=, effort_limit=)` | 输出段 | 二选一；`target`=平台 PD，`control`=直通 actuator |
| `PositionController(robot, kp=, kd=, effort_limit=, target=)` | 定值保持 | 无策略的静止/调试工程 |
| `robot.velocity_pd(target)` / `robot.effort_control(effort)` | 速度/力矩输出 | 产 native effort 信号，传 `control=` |

## 机器人标定（preset 读取，项目只引用）

`robot.policy_order` / `motor_order` / `action_scale` / `default_angles` / `kp` / `kd` /
`effort_limit`。preset 缺字段直接报错指名字段，无静默兜底。

## 其余工具（纯 Python，构图期用）

`require_vector`、`clip`、`reorder`、`affine`、`quaternion_yaw`、`inverse_rotate`、
`projected_gravity`（四元数版）、`waypoint_tracking`、`policy_inputs`（policy_bank 低层
拆分）、`Frame`、`HistoryBuffer`、`Controller`（自定义基类）、`Simulation`。

---

完整语义和示例见 BotLab 仓库 `docs/api/` 与 `docs/examples/`：
[controller.md](https://github.com/OpenLegged/BotLab) 是 Signal/状态机/policy_bank 的
权威契约，[multi-policy.md](https://github.com/OpenLegged/BotLab) 是多策略完整教程。

## Named option parameters

`project(parameters={"mode": {"label": "Mode", "default": 0,
"options": [{"label": "Stand", "value": 0}, {"label": "Walk", "value": 2}]}})`
creates a dropdown in compatible browser runtimes. Option values are unique safe
integers, labels are non-empty, and the default must belong to the list. Do not
combine `options` with `range`, `step`, or `unit`. Existing range parameters
remain sliders. Native `run.set_parameter()` rejects non-member values.

The control code still decides how a selection changes behavior. A
`Signal.action(modify=...)` hook may call `sim.reset_to()` when the selection
changes. In the native single-plan runner this reset is applied between plan
execution and physics: the transitional action is discarded, history and last
action are cleared, and the next action is recomputed at the reset frame. A
controller that requests reset again on that first frame is rejected to prevent
an endless reset loop. Native custom-controller/group reset behavior is unchanged.

For exportable motion switching, prefer the compiled declaration:
`PolicyController(robot, control=control, reset_on_change={"parameter": "mode",
"motions": {0: stand_motion, 2: walk_motion}})`. Every option value must map to a
Motion. The single-controller rule resets to frame zero with root body channel
zero and native joint ordering; it supports editor, standalone browser and native
execution without Python callbacks. It cannot be combined with `action(modify=...)`.

### Independent camera field of view

A `Sensor` preset/config accepts optional `horizontalFov` in degrees (finite,
strictly between 0 and 180). `fov` remains the vertical FoV of the full sensor.
Omitted/null `horizontalFov` preserves the aspect-derived horizontal view.
For example, `fov: 58`, `horizontalFov: 87`, `fullWidth: 48`, `fullHeight: 27`
calibrates the two axes independently, before applying `cropX`/`cropY` and
`width`/`height`. Use a matching browser runtime for editor and published sites.


## 有限时长演示与重播

`PolicyController(..., end_after_steps=500)` 在提交第 500 个控制步（含物理子步）后结束。
也可以写 `end_after_steps={"parameter": "motion_id", "steps": {0: 417, 1: 523}}`，
映射必须覆盖该选项参数的所有值。步数是正安全整数；秒数由项目显式乘控制频率并向上取整。
每次复位后的第一控制步读取所选步数，暂停和续播不重置预算。配合已有
`reset_on_change={"parameter": "motion_id", "motions": ...}`，动作切换会复位并开始新一轮。

`open_headless()` 返回值的 `completed` 表示已达到声明的结束条件；`steps(count)`
到点提前返回，结束后 `step()` 报错，调用 `reset()` 才能重播。原生 `run()` 自动返回。
编辑器和静态演示显示“已结束”，点击播放自动复位后重播。未声明时继续原有连续运行行为。
此阶段只支持单个无 Python action modifier 的 `PolicyController`；不推断任务成功、失败或跌倒。
发布包含该字段的工程时，BotLab 浏览器运行时与 Python SDK 必须一起更新到支持该字段的版本。
