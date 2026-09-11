# 选型决策表（需求 → 组件）

写 `controller.py` 前对着这张表选组件。第一列覆盖实践中绝大多数需求；
不在表里的需求先看「判断法」，再考虑逃生口。

## 分支与逻辑

| 需求 | 用什么 | 不要用 |
|------|--------|--------|
| 每帧按条件二选一（便宜分支） | `where(cond, a, b)` | — |
| 每帧按条件二选一（昂贵分支） | `mode_select` + 状态 | `where`（两边都执行） |
| 按键触发技能切换 | `commands(mappings=...)` + `state_machine` | Python `if`（构图期只跑一次） |
| 翻倒/姿态自动恢复 | `state_machine` 的 `when` 转移 | — |
| 定时返回（N 步后切状态） | `after_steps=N` 转移 | `frame.index` 手算（可行但脆） |
| 周期性动作（步态相位） | `phase_clock(machine.steps, period_steps=)` | — |
| 多个布尔条件的组合 | `&`/`\|`/`~` 或 `all_of`/`any_of` | — |
| 阈值分类为整数 | 比较 + `rule_index` | — |

## 观测

| 需求 | 用什么 |
|------|--------|
| 标准姿态观测（ang vel + gravity + dof pos/vel + last action + command） | 各观测信号 + `concat`，顺序照抄训练侧 |
| 堆叠历史帧（frame stack） | `history(src, n, layout=)`——layout 必须匹配训练惯例 |
| 减去默认关节位姿 | `robot.joint_position(default=robot.default_angles)`（**不会自动减**） |
| 观测缩放（训练时乘了系数） | `scale=` 参数原样写 |
| 深度图入观测 | `sensor.depth_frame()` + `delay(..., age=frame.age_steps)` + `resize` |
| 读平台没有命名的状态 | `robot.mujoco("sensor.data", size, target=, start=)` |
| 指令向量宽度不齐 | `concat` 补 `constant(0, size=k)`（microduck：3 速度 + 10 零） |

## 策略与输出

| 需求 | 用什么 |
|------|--------|
| 单策略 | `actor(obs).action()` → `PolicyController(robot, target=)` |
| 多策略、同一观测 | `policy_bank(inputs={key: obs})` 低层写法 |
| 多策略、观测不同 | `policy_bank(sources=..., fields=...)` 高层写法 |
| encoder→actor 流水线 | policy 的 `pipeline=[{...}, ...]` |
| MJCF 自带 position actuator | `PolicyController(robot, control=目标)`（PD 在 MJCF 里） |
| 平台 PD（preset kp/kd） | `PolicyController(robot, target=目标)` |
| 速度/力矩控制 | `robot.velocity_pd` / `robot.effort_control` + `control=` |
| 静止/调试工程 | `PositionController` |

## 判断法：这行代码该落在哪一层

对每一行 Python 问一个问题——**「它的结果每帧都变，还是整个运行期定一次？」**

- **定一次** → 普通 Python 随便写（`for` 生成 7 个 Model、字典推导转移表、numpy 算增益），
  只要末端落在 Signal 表达式上。
- **每帧变且是数据变换** → 上表选节点。
- **每帧变且是分支/循环** → 控制流组件（state_machine / mode_select / where / phase_clock）。
- **每帧变且白名单拼不出来** → 逃生口（见下），但要接受其限制。

## 逃生口与限制（CLI 语境）

| 逃生口 | 能力 | `botlab run` | `botlab build` |
|--------|------|--------------|----------------|
| `Signal.action(modify=fn)` | 每帧一个任意 Python 回调（fn 可 async） | ✅ 浏览器内 Pyodide 运行 | ✅ 产「源码+manifest」包（任何 embed 运行时可跑） |
| 自定义 `Controller.step()` | 整控制器走 Python | ✅ 同上（Pyodide 形态） | ✅ 同上 |

两种逃生口的包都**没有 compiledController**：运行时要在浏览器里起 Pyodide 加载
Python stdlib（首次多几秒、内存多几十 MB），每控制步一次回调边界。发布给大众的
demo 优先降级成图节点；逃生口留给开发期和确实无法声明式表达的逻辑。

优先级永远是：**能用图节点就不用逃生口**。

## 常见错误模式（写之前看一眼）

- 一个 plan 写了两个 `Signal.action()` → 报错；只留一个
- `signal[0]`（单下标）→ 报错；用 `signal.at(0)` 返回标量 Signal
- `history` 不写 `layout` → 报错；训练用 IsaacGym 写 `by-frame`，IsaacLab 写 `by-term`
- `where` 两边宽度不同 → 编译期报错；先 `constant` 补齐
- 参数没写 `range` 却用 `bindings` → 启动报错；交互轴必须有 `range`
- `mode_select` 分支宽度不一 → 编译期报错
- 模型输出直接当 motor 目标（忘记 `motor_order` 重排和 `action_scale` 仿射）→ 不报错但
  行为诡异；输出段必须 `default_angles + action_scale * action[motor_order]`
