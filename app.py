import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os
import onnxruntime as ort

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

    # Replace TensorFlow preprocessing
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0   # MobileNetV2 normalization

    image = np.expand_dims(image, axis=0)

    return image

# ================= PREDICTION =================
def predict(image):
    processed = preprocess_image(image)

    inputs = {model.get_inputs()[0].name: processed}
    outputs = model.run(None, inputs)

    predictions = outputs[0][0]

    class_idx = np.argmax(predictions)
    confidence = float(np.clip(predictions[class_idx], 0, 0.999))

    return CLASS_NAMES[class_idx], confidence

def generate_heatmap(image):
    image = np.array(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed

# ================= UI =================
st.title("🧠 Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original MRI")

    with col2:
        heatmap = generate_heatmap(image)
        st.image(heatmap, caption="Model Focus (Heatmap)")

    label, confidence = predict(image)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence*100:.2f}%")
