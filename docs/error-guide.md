# 报错对照（错误 → 原因 → 修复）

CLI 的错误全部是硬失败：缺什么、哪条契约破了、字段名是什么都会写明。
按错误信息关键词查这张表。

## CLI 打包期（botlab run / build / validate）

| 报错关键词 | 原因 | 修复 |
|-----------|------|------|
| `A BotLab project requires "main.py"` / `"controller.py"` | 工程缺入口文件 | 按两文件契约补齐 |
| `standard main.py entrypoint` | main.py 不是标准四行 | 照抄 README 的四行模板 |
| `legacy browser format` | 存在非空 `botlab.yaml` | 改写为 Python 声明工程（YAML 是旧格式） |
| `Resources can only be declared while a project is being applied` | main.py 阶段调用了 `Robot(...)` 等构造器 | 资源声明只能在 controller.py 模块顶层 |
| `project() can only be called once` | 两个模块都调了 `project()` | 只在 controller.py 顶层调一次 |
| `Unknown BotLab resource preset` | preset 名既不是文件也不是已知内置 | 检查 `*.preset.yaml` 文件名与 `preset=` 一致 |
| `must be a flat "<name>.preset.yaml"` | preset 路径带目录 | preset 文件放工程根，扁平命名 |
| `needs a positive integer actionDimension` | robot 无 preset 或 config 缺字段 | 补 preset：actionDimension/jointQposOffset/jointQvelOffset |
| `physics_hz must be an integer multiple of control_hz` | 频率比非整数 | 如 50/200 合法，60/200 不合法 |
| `Parameter "x" has unknown field` | 参数声明带未知键 | 参数只有 label/default/range/step/unit |
| `default is outside its range` | 参数默认值出界 | 修 default 或 range |
| `does not exist in the project` | 资源 source 指向不存在的文件 | 核对工程内相对路径 |
| `must be a URL or a safe project-relative path` | source 带 `..`、绝对路径或反斜杠 | 改为工程相对路径或 URL |
| `main.py did not register a controller` | create_controller 没返回 PolicyController | 检查返回值与四行 main |
| `control plan needs exactly one Signal.action()` | 0 个或 ≥2 个 action 标记 | 全计划恰好一个 `.action()` |
| `requires unsupported node type` | plan 里有引擎白名单外节点 | SDK 版本与 BotLab 构造版本不匹配，重同步 UI/SDK |
| `compiled controller does not match the current manifest` | 改了源码但 digest 过期 | 重新 build（正常流程不会触发） |
| `UI snapshot is missing embed.html` | 包内无 UI 快照 | `botlab ui sync --botlab <BotLab 路径>` |
| `UI snapshot is missing index.html under …/botlab/ui` | 源码 editable 安装：`ui/` 是 gitignore 的生成产物，新 clone 里没有 | 改用 `pip install botlab-python`（wheel 自带快照）；源码开发则先在 BotLab 仓 `npm run build`，再 `botlab ui sync --botlab <BotLab 路径>` |

## SDK 构图期（import controller 时）

| 报错关键词 | 原因 | 修复 |
|-----------|------|------|
| `must be a BotLab Signal` | 普通数值传进了 Signal 参数位 | 需要常量时先 `constant(value, size=)` |
| `subscript must be a sequence` | `signal[0]` 单下标 | `signal.at(0)`；gather 用 `signal[[0, 2]]` |
| `history layout must be ...` | layout 缺失或拼错 | `by-frame`（IsaacGym）/ `by-term`（IsaacLab） |
| `dimensions must match` / 宽度类报错 | concat/mode_select/where 分支宽度不一致 | 用 constant 补齐或修正拼接顺序 |
| `commands binding ... outside N parameters` | bindings 轴号越界 | 轴索引 < len(names) |
| `commands bindings require parameter ranges` | 交互轴参数无 range | project 参数补 `range` |
| `duplicated`（key/键/节点 id） | 重名 | 键、指令键位、节点 id 必须唯一 |
| `policy_bank policy "x" has no matching input` | inputs 键和 policies key 不齐 | 两边键集合一致 |
| `must define exactly one of model or pipeline` | 策略表同时有/都没有 | 二选一 |

## 浏览器运行期（embed 页面红条）

| 报错关键词 | 原因 | 修复 |
|-----------|------|------|
| `Failed to fetch release.zip` | 服务没起/被代理拦 | 保留 CLI 起的服务；代理需放行 `/release.zip` |
| `project digest does not match` | 包内容与 manifest 不一致 | 重新 `botlab build`（通常是包被改过） |
| `engine ABI` | UI 快照与 plan 编译期版本差代 | `botlab ui sync` 重新同步快照 |
| `failed SHA-256 integrity` | 传输损坏 | 检查代理是否改写字节（gzip 透明解压等） |
| 仿真黑屏 + 控制台 WebGL 报错 | 浏览器无硬件 WebGL | 换 Chrome/Edge，查 `chrome://gpu` |

## 仍然报错时

1. `botlab validate` 单独跑，拿到最早的失败点（错误按声明顺序抛出，修第一个）。
2. 读 `release.json` 里的 `runtime.requiredNodeTypes`，与组件表对照定位节点。
3. BotLab 平台编辑器里打开同构工程 Apply——平台报错带行号，能定位到 controller.py 具体位置。
