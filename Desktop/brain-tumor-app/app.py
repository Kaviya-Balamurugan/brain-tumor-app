import streamlit as st
import requests
from PIL import Image
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# ================= CONFIG =================
API_URL = "http://127.0.0.1:8000/predict"

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ================= PDF REPORT =================
def generate_pdf(prediction, confidence, level, patient_name):

    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    # ================= TITLE =================
    content.append(Paragraph("Brain Tumor Detection Report", styles['Title']))
    content.append(Spacer(1, 20))

    # ================= PATIENT INFO =================
    content.append(Paragraph(f"Patient Name: {patient_name}", styles['Normal']))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    content.append(Spacer(1, 20))

    # ================= DIAGNOSIS =================
    if confidence < 0.6:
        diagnosis = "Uncertain prediction. Confidence too low for reliable diagnosis."
    elif prediction == "no_tumor":
        diagnosis = "No Tumor Detected"
    else:
        diagnosis = f"Tumor Detected: {prediction.replace('_',' ').title()}"

    content.append(Paragraph("Diagnosis:", styles['Heading2']))
    content.append(Paragraph(diagnosis, styles['Normal']))
    content.append(Spacer(1, 15))

    # ================= CONFIDENCE =================
    content.append(Paragraph("Confidence Analysis:", styles['Heading2']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))
    content.append(Paragraph(f"Confidence Level: {level}", styles['Normal']))
    content.append(Spacer(1, 15))

    # ================= INTERPRETATION =================
    content.append(Paragraph("Interpretation:", styles['Heading2']))

    if level == "High":
        interpretation = "High confidence prediction. Likely accurate."
    elif level == "Medium":
        interpretation = "Moderate confidence. Further medical evaluation recommended."
    else:
        interpretation = "The model is uncertain about this prediction. A medical expert should review this case."

    content.append(Paragraph(interpretation, styles['Normal']))
    content.append(Spacer(1, 20))

    # ================= DISCLAIMER =================
    content.append(Paragraph("Disclaimer:", styles['Heading2']))
    content.append(Paragraph(
        "This report is generated using an AI model and is intended for educational purposes only. "
        "It should not be used as a substitute for professional medical advice.",
        styles['Normal']
    ))

    doc.build(content)

    return file_path

# ================= API CALL =================
def predict_from_api(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    files = {"file": ("image.jpg", buf.getvalue(), "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files)
        return response.json()
    except:
        return {"error": "API not running"}

# ================= UI =================
st.title("🧠 Brain Tumor Detection System")

st.markdown("""
### 🔬 AI-powered MRI Analysis Tool  
Detect tumor types using deep learning

- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  
""")
patient_name = st.text_input("Enter Patient Name", "Anonymous")
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Uploaded MRI", use_container_width=True)

    with col2:
        st.subheader("🧪 Analysis Result")

        with st.spinner("Analyzing MRI..."):
            result = predict_from_api(image)

        # ================= ERROR HANDLING =================
        if "error" in result:
            st.error("❌ API not running. Start FastAPI server.")
        
        elif "detail" in result:
            st.error(f"❌ {result['detail']}")

        else:
            label = result["prediction"]
            confidence = result["confidence"]
            level = result["confidence_level"]

            # ================= RESULT =================
            if label == "no_tumor":
                st.success("✅ No Tumor Detected")
            else:
                st.error(f"⚠️ Tumor Detected: {label.replace('_',' ').title()}")

            st.info(f"📊 Confidence: {confidence*100:.2f}%")
            st.write(f"🔍 Confidence Level: **{level}**")

            # ================= INTERPRETATION =================
            st.subheader("🩺 Interpretation")

            if level == "High":
                st.write("High confidence prediction. Likely accurate.")
            elif level == "Medium":
                st.write("Moderate confidence. Further medical review recommended.")
            else:
                st.warning("Low confidence. Please consult a specialist.")

            # ================= PROBABILITY =================
            if "all_probabilities" in result:
                st.markdown("---")
                st.subheader("📊 Prediction Distribution")

                for k, v in result["all_probabilities"].items():
                    st.progress(float(v))
                    st.write(f"{k}: {v*100:.2f}%")

            # ================= PDF =================
            pdf = generate_pdf(label, confidence, level, patient_name)

            with open(pdf, "rb") as f:
                st.download_button("📄 Download Report", f, file_name="report.pdf")

# ================= FOOTER =================
st.markdown("---")
st.caption("👩‍💻 Developed by Kaviya")
st.caption("⚠️ This tool is for educational purposes only and not for medical diagnosis.")