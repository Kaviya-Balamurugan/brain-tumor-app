import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os
import onnxruntime as ort
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

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
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)

    return image

# ================= PREDICTION =================
def predict(image):
    processed = preprocess_image(image)
    inputs = {model.get_inputs()[0].name: processed}
    outputs = model.run(None, inputs)
    return outputs[0][0]

# ================= PDF REPORT =================
def generate_pdf(prediction, confidence):
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Brain Tumor Detection Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))
    content.append(Paragraph(f"Date: {datetime.now()}", styles['Normal']))

    doc.build(content)
    return file_path

# ================= UI =================
st.title("🧠 Brain Tumor Classification System")

st.markdown("""
### 🔬 AI-powered MRI Analysis Tool  
Detects brain tumor types using deep learning (MobileNetV2 + ONNX)

- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  
""")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Uploaded MRI", use_container_width=True)

    # Prediction
    predictions = predict(image)
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    label = CLASS_NAMES[class_idx]

    with col2:
        st.subheader("🧪 Analysis Result")

        # SMART RESULT
        if label == "no_tumor":
            st.success("✅ No Tumor Detected")
        else:
            st.error(f"⚠️ Tumor Detected: {label.replace('_',' ').title()}")

        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # INTERPRETATION
        st.subheader("🩺 Interpretation")

        if confidence > 0.85:
            st.write("High confidence prediction. Likely accurate.")
        elif confidence > 0.6:
            st.write("Moderate confidence. Further medical review recommended.")
        else:
            st.warning("Low confidence. Please consult a specialist.")

    # ================= DISTRIBUTION =================
    st.markdown("---")
    st.subheader("📊 Confidence Distribution")

    for i, prob in enumerate(predictions):
        st.progress(float(prob))
        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

    # ================= TOP 2 =================
    st.subheader("🔍 Top Predictions")

    top_indices = predictions.argsort()[-2:][::-1]

    for i in top_indices:
        st.write(f"👉 {CLASS_NAMES[i]}: {predictions[i]*100:.2f}%")

    # ================= PDF DOWNLOAD =================
    pdf = generate_pdf(label, confidence)

    with open(pdf, "rb") as f:
        st.download_button("📄 Download Report", f, file_name="report.pdf")

    # ================= MODEL INFO =================
    with st.expander("ℹ️ Model Information"):
        st.write("""
        - Model: MobileNetV2
        - Framework: ONNX Runtime
        - Input Size: 224x224
        - Classes: 4
        - Accuracy: ~91%
        """)

    # ================= ABOUT =================
    with st.expander("📘 About this Project"):
        st.write("""
        This project uses deep learning to classify brain tumors from MRI scans.
        The model was trained using TensorFlow and deployed using ONNX Runtime
        for efficient and scalable inference.

        Technologies:
        - TensorFlow (training)
        - ONNX Runtime (deployment)
        - Streamlit (UI)
        """)

# ================= FOOTER =================
st.markdown("---")
st.caption("👩‍💻 Developed by Kaviya")
st.caption("⚠️ This tool is for educational purposes only and not for medical diagnosis.")
