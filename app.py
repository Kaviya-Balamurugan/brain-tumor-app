import streamlit as st
import requests
from PIL import Image
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# 🔥 IMPORTANT: PUT YOUR REAL API LINK
API_URL = "https://your-api.onrender.com/predict"

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ================= PDF =================
def generate_pdf(prediction, confidence, level, patient_name):
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Brain Tumor Detection Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Patient Name: {patient_name}", styles['Normal']))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    content.append(Spacer(1, 20))

    if confidence < 0.6:
        diagnosis = "Uncertain prediction. Confidence too low."
    elif prediction == "no_tumor":
        diagnosis = "No Tumor Detected"
    else:
        diagnosis = f"Tumor Detected: {prediction.replace('_',' ').title()}"

    content.append(Paragraph("Diagnosis:", styles['Heading2']))
    content.append(Paragraph(diagnosis, styles['Normal']))
    content.append(Spacer(1, 15))

    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))
    content.append(Paragraph(f"Confidence Level: {level}", styles['Normal']))

    doc.build(content)
    return file_path

# ================= API CALL =================
def predict_from_api(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    files = {"file": ("image.jpg", buf.getvalue(), "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files, timeout=60)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ================= UI =================
st.title("🧠 Brain Tumor Detection")

patient_name = st.text_input("Enter Patient Name", "Anonymous")
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, width="stretch")

    with col2:
        with st.spinner("Analyzing MRI..."):
            result = predict_from_api(image)

        if "error" in result:
            st.error(result["error"])
        else:
            label = result["prediction"]
            confidence = result["confidence"]
            level = result["confidence_level"]

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence*100:.2f}%")

            pdf = generate_pdf(label, confidence, level, patient_name)

            with open(pdf, "rb") as f:
                st.download_button("Download Report", f, "report.pdf")