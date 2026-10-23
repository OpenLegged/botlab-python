"""Reject external tensor files without requiring the ONNX Python compiler.

This is a narrow wire-format traversal, not a model parser or validator. ORT
validates model semantics afterwards. Field IDs follow the public ONNX schema:
https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
"""
from .protocol import HilError

# Only messages that can contain TensorProto need recursive inspection.
_CHILDREN = {
    "model": {7: "graph", 20: "training", 25: "function"},
    "training": {1: "graph", 2: "graph"},
    "function": {7: "node", 11: "attribute"},
    "graph": {1: "node", 5: "tensor", 15: "sparse"},
    "node": {5: "attribute"},
    "attribute": {5: "tensor", 6: "graph", 10: "tensor", 11: "graph",
                  22: "sparse", 23: "sparse"},
    "sparse": {1: "tensor", 2: "tensor"},
    "tensor": {},
}


def _varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            break
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise HilError("Invalid ONNX protobuf encoding")


def require_embedded_tensors(content: bytes):
    def visit(data, kind, depth):
        if depth > 64:
            raise HilError("ONNX nesting exceeds the supported limit")
        position = 0
        while position < len(data):
            key, position = _varint(data, position)
            field, wire = key >> 3, key & 7
            if not field:
                raise HilError("Invalid ONNX protobuf field")
            if kind == "tensor" and field == 13:
                raise HilError("External ONNX tensor files are not supported; embed all weights")
            if wire == 0:
                value, position = _varint(data, position)
                if kind == "tensor" and field == 14 and value != 0:
                    raise HilError("External ONNX tensor files are not supported; embed all weights")
            elif wire == 2:
                length, position = _varint(data, position)
                end = position + length
                if end > len(data):
                    raise HilError("Truncated ONNX protobuf field")
                child = _CHILDREN[kind].get(field)
                if child:
                    visit(data[position:end], child, depth + 1)
                position = end
            elif wire in (1, 5):
                position += 8 if wire == 1 else 4
                if position > len(data):
                    raise HilError("Truncated ONNX protobuf field")
            else:
                raise HilError("Unsupported ONNX protobuf wire type")
    visit(memoryview(content), "model", 0)
