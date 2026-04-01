from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import io
import gdown
import os

app = FastAPI(title="Brain Tumor Detection API")

# ================= DOWNLOAD MODEL =================
MODEL_PATH = "model.onnx"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1fbTb-NEivEEY4-OzmNx5HQYokG6CRd3L"
    gdown.download(url, MODEL_PATH, quiet=False)

# ================= LOAD MODEL =================
try:
    model = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

IMAGE_SIZE = 224
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ================= HEALTH CHECK =================
@app.get("/")
def home():
    return {"message": "Brain Tumor Classification API is running 🚀"}

# ================= PREPROCESS =================
def preprocess_image(image):
    image = np.array(image)

    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)

    return image

# ================= API ENDPOINT =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Image too small")

        processed = preprocess_image(image)

        inputs = {model.get_inputs()[0].name: processed}
        outputs = model.run(None, inputs)

        predictions = outputs[0][0]

        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])

        # Confidence level
        if confidence > 0.85:
            level = "High"
        elif confidence > 0.6:
            level = "Medium"
        else:
            level = "Low"

        return {
            "prediction": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4),
            "confidence_level": level,
            "all_probabilities": {
                CLASS_NAMES[i]: float(predictions[i])
                for i in range(len(CLASS_NAMES))
            }
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
