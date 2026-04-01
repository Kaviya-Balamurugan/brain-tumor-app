# 🧠 Brain Tumor Detection System

An end-to-end AI-powered web application for detecting brain tumors from MRI images using deep learning and deploying it as a scalable web product.

---

## 🚀 Live Demo

- 🌐 Frontend (Streamlit): https://your-app.streamlit.app  
- ⚡ Backend API (FastAPI): https://your-api.onrender.com  

---

## 🔬 Overview

This project uses a deep learning model to classify brain MRI images into different tumor categories. The system is built with a modern architecture separating frontend and backend for scalability and performance.

---

## 🎯 Features

- 📤 Upload MRI images for analysis  
- 🧠 Detects 4 classes:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor  
- 📊 Confidence score with interpretation  
- 📄 Generate downloadable medical-style PDF report  
- ⚡ Fast API-based predictions  
- 🌐 Fully deployed web application  
- 🧑‍⚕️ Safe AI output (confidence-aware predictions)

---

## 🏗️ System Architecture
User (Browser)
↓
Streamlit Frontend
↓
FastAPI Backend (Render)
↓
ONNX Model (Optimized Inference)
↓
Prediction Output


---

## 🧠 Tech Stack

### 🔹 Machine Learning
- TensorFlow (Model Training)
- MobileNetV2 (Transfer Learning)

### 🔹 Model Optimization
- ONNX Runtime (Fast inference)

### 🔹 Backend
- FastAPI (REST API)
- Uvicorn (ASGI server)

### 🔹 Frontend
- Streamlit (Interactive UI)

### 🔹 Tools & Libraries
- NumPy
- OpenCV
- Pillow
- ReportLab (PDF generation)
- Requests (API communication)

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/brain-tumor-app.git
cd brain-tumor-app

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Frontend
streamlit run app.py

4️⃣ Run Backend (Optional)
uvicorn api:app --reload

📊 Model Details
Model: MobileNetV2
Input Size: 224 × 224
Classes: 4
Accuracy: ~91%
Output: Probability distribution over classes

📄 Sample Output
Prediction: Glioma Tumor
Confidence: 81.39%
Confidence Level: High
PDF Report Generated

🧪 API Usage
Endpoint:
POST /predict
Example Request:
Upload image file (multipart/form-data)
Example Response:
{
  "prediction": "glioma_tumor",
  "confidence": 0.8139,
  "confidence_level": "High",
  "all_probabilities": {
    "glioma_tumor": 0.81,
    "meningioma_tumor": 0.02,
    "no_tumor": 0.10,
    "pituitary_tumor": 0.05
  }
}

⚠️ Disclaimer

This project is intended for educational purposes only.
It should not be used for real medical diagnosis.
Always consult a qualified medical professional.

💡 Key Learnings
End-to-end ML system design
Model optimization using ONNX
API development with FastAPI
Frontend-backend integration
Real-world deployment (Render + Streamlit Cloud)
Handling production-level issues
