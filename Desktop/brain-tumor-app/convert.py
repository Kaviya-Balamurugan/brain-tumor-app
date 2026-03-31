import tensorflow as tf
import tf2onnx

# Load your model
model = tf.keras.models.load_model("brain_tumor_detector.keras")

# Define input shape
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ Conversion complete! model.onnx created")