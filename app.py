import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os
import onnxruntime as ort

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

# ================= DOWNLOAD MODEL =================
url = "https://drive.google.com/uc?id=1fbTb-NEivEEY4-OzmNx5HQYokG6CRd3L"
output = "model.onnx"

if not os.path.exists(output):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(url, output, quiet=False)

# ================= CONFIG =================
IMAGE_SIZE = 224
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return ort.InferenceSession(output)

model = load_model()

# ================= PREPROCESS =================
def preprocess_image(image):
    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0  # MobileNetV2 normalization
    image = np.expand_dims(image, axis=0)

    return image

# ================= PREDICTION =================
def predict(image):
    processed = preprocess_image(image)

    inputs = {model.get_inputs()[0].name: processed}
    outputs = model.run(None, inputs)

    predictions = outputs[0][0]
    return predictions

# ================= UI =================
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to classify tumor type using AI.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Prediction
    predictions = predict(image)

    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]

    st.success(f"🧠 Prediction: {CLASS_NAMES[class_idx]}")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")

    # ================= TOP 2 PREDICTIONS =================
    st.subheader("🔍 Top Predictions")

    top_indices = predictions.argsort()[-2:][::-1]

    for i in top_indices:
        st.write(f"👉 {CLASS_NAMES[i]}: {predictions[i]*100:.2f}%")

    # ================= CONFIDENCE DISTRIBUTION =================
    st.subheader("📊 Confidence Distribution")

    for i, prob in enumerate(predictions):
        st.progress(float(prob))
        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

    # ================= PROFESSIONAL NOTE =================
    st.warning(
        "⚠️ Grad-CAM visualization is not included in this deployed version "
        "because ONNX runtime does not support gradient-based explainability. "
        "However, it is implemented in the TensorFlow research version."
    )
