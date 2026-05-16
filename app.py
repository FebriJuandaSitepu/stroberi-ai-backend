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

    # Return JSON
    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }