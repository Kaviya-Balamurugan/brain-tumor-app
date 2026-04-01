# 🧠 Brain Tumor Detection System

An end-to-end AI-powered web application for detecting brain tumors from MRI images using deep learning.

---

## 🚀 Live Demo

- 🌐 Frontend (Streamlit): https://your-app.streamlit.app  
- ⚡ Backend API (FastAPI): https://your-api.onrender.com  

---

## 🔬 Features

- Upload MRI images for tumor detection
- Classifies:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- Confidence score with interpretation
- PDF medical-style report generation
- Real-time API-based prediction
- Clean and user-friendly UI

---

## 🧠 Tech Stack

### 🔹 Machine Learning
- TensorFlow (training)
- MobileNetV2 (transfer learning)

### 🔹 Deployment
- ONNX Runtime (optimized inference)
- FastAPI (backend API)
- Streamlit (frontend UI)

### 🔹 Tools
- OpenCV
- NumPy
- ReportLab

---

## 🏗️ Architecture

User → Streamlit → FastAPI → ONNX Model → Prediction

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/your-username/brain-tumor-app
cd brain-tumor-app
pip install -r requirements.txt
streamlit run app.py
