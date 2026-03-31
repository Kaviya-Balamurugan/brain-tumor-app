import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import gdown
import os


# ================= DOWNLOAD MODEL =================
url = "https://drive.google.com/uc?id=1fbTb-NEivEEY4-OzmNx5HQYokG6CRd3L"
output = "brain_tumor_detector.keras"

if not os.path.exists(output):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(url, output, quiet=False)

# ================= CONFIG =================
IMAGE_SIZE = 224
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(output)

model = load_model()

# ================= PREPROCESS =================
def preprocess_image(image):
    image = np.array(image)

    # Fix grayscale images
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

# ================= PREDICTION =================
def predict(image):
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]

    class_idx = np.argmax(predictions)
    confidence = float(np.clip(predictions[class_idx], 0, 0.999))

    return CLASS_NAMES[class_idx], confidence

# ================= UI =================
st.title("🧠 Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    label, confidence = predict(image)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence*100:.2f}%")
