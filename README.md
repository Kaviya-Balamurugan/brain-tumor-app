# 🧠 Brain Tumor Detection System

An end-to-end AI-powered web application that detects brain tumors from MRI images using deep learning.
The system integrates **Machine Learning, FastAPI, Streamlit, and SQLite** into a full-stack solution.

---

## 🚀 Live Demo

* 🌐 Frontend (Streamlit): https://brain-tumor-app-th2lex7glc6kubgupwet3q.streamlit.app
* ⚡ Backend API (Render): https://brain-tumor-app-1-hmhx.onrender.com

---

## 🎯 Features

* 🧠 Brain tumor classification (4 classes)
* 📤 MRI image upload
* ⚡ Fast prediction using ONNX Runtime
* 📊 Confidence score & interpretation
* ⏱️ API response time display
* 📄 PDF report generation
* 🧪 Image quality check (brightness validation)
* 🔐 User authentication (Login / Signup)
* 🗄️ Database storage (SQLite)
* 📂 Previous reports history
* 📊 Analytics dashboard (charts & insights)
* 🎨 Professional UI with custom styling

---

## 🧠 Classes Detected

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

---

## 🏗️ System Architecture

```
User → Streamlit UI → FastAPI (Render) → ONNX Model → Response → UI → Database
```

---

## ⚙️ Tech Stack

### 🔹 Machine Learning

* TensorFlow
* MobileNetV2
* ONNX

### 🔹 Backend

* FastAPI
* ONNX Runtime
* Uvicorn

### 🔹 Frontend

* Streamlit

### 🔹 Database

* SQLite

### 🔹 Deployment

* Render (Backend)
* Streamlit Cloud (Frontend)

---

## 📂 Project Structure

```
brain-tumor-app/
│
├── app.py               # Streamlit frontend
├── api.py               # FastAPI backend
├── database.py          # Database functions
├── model.onnx           # Trained model
├── requirements.txt     # Dependencies
├── start.sh             # Render startup script
└── README.md
```

---

## 🔧 Installation (Local Setup)

### 1. Clone the repository

```
git clone https://github.com/your-username/brain-tumor-app.git
cd brain-tumor-app
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### 🔹 Start Backend (FastAPI)

```
uvicorn api:app --reload
```

### 🔹 Start Frontend (Streamlit)

```
streamlit run app.py
```

---

## 🌐 API Endpoint

### POST `/predict`

**Input:** MRI Image
**Output:**

```json
{
  "prediction": "glioma_tumor",
  "confidence": 0.87,
  "confidence_level": "High"
}
```

---

## 📊 Key Features Explained

### 🔐 Authentication

* Signup & login system using SQLite
* Session management using Streamlit

### 📂 Report Storage

* Stores predictions with patient name & timestamp
* Prevents duplicate entries

### 📊 Analytics Dashboard

* Total scans
* Tumor distribution
* Confidence trends

### ⚠️ Image Validation

* Detects poor-quality MRI images (too dark/bright)

---

## 🧠 Model Details

* Architecture: MobileNetV2
* Input Size: 224x224
* Accuracy: ~91%
* Framework: TensorFlow → ONNX

---

## 🐛 Challenges & Solutions

| Problem              | Solution                     |
| -------------------- | ---------------------------- |
| Duplicate DB entries | Session-based deduplication  |
| API timeout          | Added request timeout        |
| Model loading error  | Auto-download model          |
| Deployment issues    | Fixed dependencies & startup |

---

## 🧠 What I Learned

* End-to-end ML system design
* API integration with frontend
* Deployment on cloud platforms
* Handling real-world bugs
* Database integration
* UI/UX improvements

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and should not be used for medical diagnosis.

---

## 👩‍💻 Author

**Kaviya**

* GitHub: [https://github.com/your-username](https://github.com/Kaviya-Balamurugan)
* LinkedIn: [https://linkedin.com/in/your-profile](https://www.linkedin.com/in/kaviyabalamurugan/)
