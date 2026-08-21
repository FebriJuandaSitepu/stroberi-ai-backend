from fastapi import FastAPI, UploadFile, File
import shutil

from predict import predict_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Batas minimum confidence supaya dianggap valid (untuk kematangan)
CONFIDENCE_THRESHOLD = 70.0  # dalam persen, silakan disesuaikan

# =====================================
# ROOT ENDPOINT
# =====================================

@app.get("/")
def home():
    return {
        "message": "API Stroberi AI Aktif"
    }

# =====================================
# PREDICT ENDPOINT
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_file = "temp.jpg"

    # Simpan file sementara
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict AI
    result, confidence = predict_image(temp_file)
    confidence = round(confidence, 2)

    # =====================================
    # CEK 1: GATEKEEPER - apakah objeknya stroberi?
    # =====================================
    if result == "Bukan Stroberi":
        return {
           "prediction": "Tidak Dikenali",
            "confidence": confidence,
            "is_valid": False,
            "message": "Gambar tidak terdeteksi sebagai stroberi. Silakan upload gambar stroberi yang lebih jelas."
        }

    # =====================================
    # CEK 2: THRESHOLD - apakah AI cukup yakin dengan kematangannya?
    # =====================================
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Tidak Dikenali",
            "confidence": confidence,
            "is_valid": False,
            "message": "AI kurang yakin dengan tingkat kematangan gambar ini. Coba upload gambar yang lebih jelas."
        }

    # =====================================
    # Return JSON normal
    # =====================================
    return {
        "prediction": result,
        "confidence": confidence,
        "is_valid": True
    }