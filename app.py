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

if "last_entry" not in st.session_state:
    st.session_state.last_entry = None

# ================= INIT =================
create_table()
create_user_table()

API_URL = "https://brain-tumor-app-1-hmhx.onrender.com/predict"

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ================= UI STYLING =================
st.markdown("""
<style>
.main {background-color: #f5f7fb;}
h1 {color: #1f2c56; font-weight: 700;}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
.success-box {background: #e6f9f0; padding: 15px; border-radius: 10px;}
.warning-box {background: #fff4e5; padding: 15px; border-radius: 10px;}
.error-box {background: #ffe6e6; padding: 15px; border-radius: 10px;}
section[data-testid="stSidebar"] {background-color: #1f2c56; color: white;}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
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

    else:
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

# ================= HEADER =================
st.markdown("""
<h1>🧠 Brain Tumor Detection System</h1>
<p style='color:gray;'>AI-powered MRI analysis for tumor classification</p>
""", unsafe_allow_html=True)

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
    content.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))
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

# ================= MAIN =================
patient_name = st.text_input("Enter Patient Name", st.session_state.username)
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    quality_status = check_image_quality(image)
    if quality_status != "good":
        st.warning(quality_status)

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

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            current_entry = (patient_name, label, round(confidence, 4), current_time)

            if st.session_state.last_entry != current_entry:
                insert_report(patient_name, label, confidence, current_time)
                st.session_state.last_entry = current_entry

            # ===== RESULT CARD =====
            st.markdown('<div class="card">', unsafe_allow_html=True)

            if confidence < 0.6:
                st.markdown('<div class="warning-box">⚠️ Uncertain Prediction</div>', unsafe_allow_html=True)
            elif label == "no_tumor":
                st.markdown('<div class="success-box">✅ No Tumor Detected</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box">🚨 Tumor Detected: {label}</div>', unsafe_allow_html=True)

            st.markdown(f"### 📊 Confidence: {confidence*100:.2f}%")
            st.progress(float(confidence))

            if "response_time" in result:
                st.caption(f"⏱️ Prediction Time: {result['response_time']} sec")

            st.markdown('</div>', unsafe_allow_html=True)

            pdf = generate_pdf(label, confidence, "", patient_name)
            with open(pdf, "rb") as f:
                st.download_button("📄 Download Report", f, "report.pdf")

# ================= REPORTS =================
st.markdown("---")
st.subheader("📂 Previous Reports")

reports = get_reports()
unique_reports = list(dict.fromkeys(reports))

for r in unique_reports[:10]:
    st.markdown(f"""
    <div class="card">
    👤 <b>{r[1]}</b><br>
    🧠 {r[2]}<br>
    📊 {r[3]*100:.2f}%<br>
    🕒 {r[4]}
    </div>
    """, unsafe_allow_html=True)

# ================= ANALYTICS =================
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

if unique_reports:
    df = pd.DataFrame(unique_reports, columns=["id","name","prediction","confidence","date"])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.bar_chart(df["prediction"].value_counts())
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.line_chart(df["confidence"])
    st.markdown('</div>', unsafe_allow_html=True)