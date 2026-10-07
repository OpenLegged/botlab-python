# Native runtime fixtures

`processors.json` is copied byte for byte from BotLab's
`src/domain/python/golden/processors.json`, exported by
`processorGoldenExport.test.ts`. Both repositories replay these traces to check
that native and browser processors preserve the same numerical behavior.

The ONNX files in `models/` are frozen copies of BotLab's
`public/templates/go2w-lab2mj/assets/models/vtm/student.onnx` and
`public/templates/go2w-lab2mj/assets/models/vtm_gru_sru/student_memory.onnx`.
They retain the native inference and recurrent-state tests without requiring a
BotLab checkout. These are test assets and are excluded from the Python wheel.
