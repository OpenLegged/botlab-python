# ONNX 策略导出指南

把训练框架里的策略导出成 BotLab 能在浏览器里推理的单文件 ONNX。围绕三个契约：**单文件**、
**一个主输入**、**图内名字与声明一致**。`botlab build` 打包时(装了 `onnx` 包的话)会逐项校验，
但导出时就按下面的清单做，基本不会留到打包期。

## 导出清单

1. **单文件**。ONNX 的权重默认内嵌在模型文件里，但 torch 2.x 的 dynamo 导出器(以及
   `onnx.save_model(..., save_as_external_data=True)`)会把权重写到旁边的 `.data` 文件。
   这种模型在导出目录加载一切正常，发布后必炸——浏览器拿到的是包内的模型字节，旁边没有那个
   文件。内联回去：

   ```python
   import onnx

   model = onnx.load(path, load_external_data=False)
   onnx.external_data_helper.convert_model_from_external_data(model)
   onnx.save(model, path)
   ```

2. **一个主输入**。`Model(...)(observation, input_name=..., output_name=...)` 把拼接后的
   观测喂给图的**主输入**；多输入模型用 `constants=` 补静态输入。导出时把策略包成
   单输入 `nn.Module`（内部再切分拼接）最省事：

   ```python
   class SingleInput(torch.nn.Module):
       def __init__(self, actor):
           super().__init__()
           self.actor = actor

       def forward(self, obs):          # (1, obs_dim) → (1, action_dim)
           batch = split_observation(obs)   # 训练侧同一套切分逻辑
           return self.actor(batch)
   ```

3. **名字一致**。`input_name="obs"` / `output_name="action"` 是图里 `graph.input` /
   `graph.output` 的名字，`torch.onnx.export(..., input_names=["obs"], output_names=["action"])`
   直接控制它们。名字对不上是运行期的静默错位或深层报错，打包校验会直接拒。

4. **形状**。固定形状(如 `(1, obs_dim)`)最稳;动态维度用 `dynamic_axes` 声明,但输出宽度
   要能推断(静态维,或在 `Model(...)(..., output_dim=N)` 里显式给定),否则引擎报
   `Cannot infer outputDim`。

5. **奇偶校验**。导出脚本内置 torch↔onnxruntime 随机输入对比,阈值建议 `max|Δ| < 1e-4`:

   ```python
   import numpy as np, onnxruntime as ort, torch

   session = ort.InferenceSession(path)
   for seed in range(8):
       obs = torch.randn(1, obs_dim, generator=torch.Generator().manual_seed(seed))
       expected = torch_model(obs).detach().numpy()
       got = session.run(None, {"obs": obs.numpy()})[0]
       assert np.abs(expected - got).max() < 1e-4
   ```

   首样本用真实首帧观测(如 `[0, 0, -1]` 的重力)更接近部署分布。

## 打包期校验

`create_release_package`(即 `botlab build` / `botlab run`)在装了 `onnx` 包时对每个
model 资产做三项检查,报错带修复建议:

| 检查 | 报错关键词 |
|------|-----------|
| 文件可解析 | `not a valid ONNX model` |
| 权重内嵌 | `stores weights in external file(s)` |
| 名字一致 | `call declares input_name/output_name ... but the model ... are [...]` |

没装 `onnx` 时打包信任字节(保持旧行为);装上即生效:`pip install onnx`,或用本仓库的
`[dev]` extra。校验失败的第一行永远是资源 id,照单修文件即可。

## 参考实现

`casbot-project/tools/export_policy.py`(GMTrack CASBOT02,1548 维观测的 transformer actor)
是按本清单落地的完整例子:单输入包装、opset 17、固定形状、自动内联外部权重、8 样本奇偶
校验(实测 1.9e-06)。
