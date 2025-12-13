import onnx, onnxruntime as ort, numpy as np

m = onnx.load("runs/deit_tiny.onnx"); onnx.checker.check_model(m)
print("ONNX ok. IR:", m.ir_version, "Opset:", [opset.version for opset in m.opset_import])

sess = ort.InferenceSession("runs/deit_tiny.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name

H = W = 160
x = np.random.rand(1,3,H,W).astype("float32")
y, = sess.run([out], {inp: x})

print("Output shape:", y.shape, "min/max:", float(y.min()), float(y.max()))
