import streamlit as st
import requests
from PIL import Image
import io
import time
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
from database import create_table, insert_report, get_reports

# ================= INIT =================
create_table()

API_URL = "https://brain-tumor-app-1-hmhx.onrender.com/predict"

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
    content.append(Spacer(1, 15))

    if confidence < 0.6:
        interpretation = "Low confidence. Prediction is uncertain."
    elif confidence < 0.85:
        interpretation = "Moderate confidence. Further medical review recommended."
    else:
        interpretation = "High confidence. Likely accurate prediction."

    content.append(Paragraph("Interpretation:", styles['Heading2']))
    content.append(Paragraph(interpretation, styles['Normal']))

    doc.build(content)
    return file_path

# ================= IMAGE QUALITY =================
def check_image_quality(image):
    img = np.array(image)
    brightness = np.mean(img)

    if brightness < 40:
        return "⚠️ Image too dark"
    elif brightness > 220:
        return "⚠️ Image too bright"
    else:
        return "good"

# ================= API =================
def predict_from_api(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    files = {"file": ("image.jpg", buf.getvalue(), "image/jpeg")}

    try:
        start_time = time.time()
        response = requests.post(API_URL, files=files, timeout=60)
        end_time = time.time()

        result = response.json()
        result["response_time"] = round(end_time - start_time, 2)

        return result

    except Exception as e:
        return {"error": str(e)}

# ================= UI =================
st.title("🧠 Brain Tumor Classification")

patient_name = st.text_input("Enter Patient Name", "Anonymous")
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    quality_status = check_image_quality(image)
    if quality_status != "good":
        st.warning(quality_status)
        st.info("Try uploading a clearer MRI image for better accuracy.")

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

            # ✅ SAVE TO DATABASE (fixed indentation)
            insert_report(
                patient_name,
                label,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            )

            # ================= RESULT =================
            if confidence < 0.6:
                st.warning("⚠️ Uncertain Prediction")
                st.write("The model is not confident. Please consult a specialist.")

            elif confidence < 0.85:
                if label == "no_tumor":
                    st.info("🟡 Possibly No Tumor (Needs Verification)")
                else:
                    st.warning(f"⚠️ Possible Tumor: {label.replace('_',' ').title()}")

            else:
                if label == "no_tumor":
                    st.success("✅ No Tumor Detected (High Confidence)")
                else:
                    st.error(f"🚨 Tumor Detected: {label.replace('_',' ').title()}")

            st.info(f"📊 Confidence: {confidence*100:.2f}%")
            st.progress(float(confidence))

            if "response_time" in result:
                st.caption(f"⏱️ Prediction Time: {result['response_time']} sec")

            st.subheader("🩺 Interpretation")

            if confidence < 0.6:
                st.warning("Low confidence — do NOT rely on this prediction.")
            elif confidence < 0.85:
                st.write("Moderate confidence — further medical review recommended.")
            else:
                st.success("High confidence prediction — likely accurate.")

            # ================= PDF =================
            pdf = generate_pdf(label, confidence, level, patient_name)
            with open(pdf, "rb") as f:
                st.download_button("📄 Download Report", f, "report.pdf")

# ================= REPORTS =================
st.markdown("---")
st.subheader("📂 Previous Reports")

reports = get_reports()

for r in reports[:10]:
    st.write(f"👤 {r[1]} | 🧠 {r[2]} | 📊 {r[3]*100:.2f}% | 🕒 {r[4]}")

# ================= ANALYTICS =================
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

if reports:
    df = pd.DataFrame(reports, columns=["id", "name", "prediction", "confidence", "date"])

    st.write(f"📈 Total Scans: {len(df)}")

    most_common = df["prediction"].value_counts().idxmax()
    st.write(f"🧠 Most Common Prediction: {most_common}")

    st.write("📊 Tumor Distribution")
    st.bar_chart(df["prediction"].value_counts())

    st.write("📉 Confidence Trend")
    st.line_chart(df["confidence"])

else:
    st.info("No reports available yet.")