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
from database import create_user_table, add_user, verify_user

# ================= SESSION INIT =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "saved" not in st.session_state:
    st.session_state.saved = False

# ================= INIT =================
create_table()
create_user_table()

API_URL = "https://brain-tumor-app-1-hmhx.onrender.com/predict"

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ================= LOGIN SYSTEM =================
menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

if not st.session_state.logged_in:

    if menu == "Login":
        st.title("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = verify_user(username, password)

            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    elif menu == "Signup":
        st.title("📝 Signup")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Signup"):
            if add_user(new_user, new_pass):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")

    st.stop()

# ================= SIDEBAR =================
st.sidebar.write(f"👤 {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

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

# ================= MAIN UI =================
st.title("🧠 Brain Tumor Classification")

patient_name = st.text_input("Enter Patient Name", st.session_state.username)
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    # 🔥 reset save flag for new upload
    st.session_state.saved = False

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

            # ✅ SAVE ONLY ONCE
            if not st.session_state.saved:
                insert_report(
                    patient_name,
                    label,
                    confidence,
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                )
                st.session_state.saved = True

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence*100:.2f}%")

            if "response_time" in result:
                st.caption(f"⏱️ Prediction Time: {result['response_time']} sec")

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

    st.bar_chart(df["prediction"].value_counts())
    st.line_chart(df["confidence"])

else:
    st.info("No reports available yet.")