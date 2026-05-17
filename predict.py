import torch
import torch.nn as nn
import joblib
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# =====================================
# LOAD LABEL CLASS
# =====================================

with open("model/classes.txt", "r") as f:
    classes = [x.strip() for x in f.readlines()]

# =====================================
# LOAD CNN MODEL (MobileNetV3)
# =====================================

model = mobilenet_v3_small()

# Ubah classifier menjadi 3 kelas
model.classifier[3] = nn.Linear(1024, 3)

# Load trained model
state_dict = torch.load(
    "model/cnn_feature_extractor_stroberi.pth",
    map_location="cpu"
)

model.load_state_dict(
    state_dict,
    strict=False
)

model.eval()

print("✅ MobileNetV3 berhasil diload")

# =====================================
# LOAD SVM MODEL
# =====================================

svm_model = joblib.load(
    "model/svm_classifier_stroberi.pkl"
)

print("✅ SVM berhasil diload")

# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================================
# PREDICT FUNCTION
# =====================================

def predict_image(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform image
    image_tensor = transform(image)

    # Tambahkan batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # =====================================
    # FEATURE EXTRACTION CNN
    # =====================================

    with torch.no_grad():

        # Ambil feature map dari MobileNetV3
        features = model.features(image_tensor)

        # Pooling
        features = torch.nn.functional.adaptive_avg_pool2d(
            features,
            (1, 1)
        )

        # Flatten menjadi vector
        features = torch.flatten(features, 1)

    # Convert tensor ke numpy
    features = features.numpy()

    # Debug shape
    print("📌 Shape fitur:", features.shape)

    # =====================================
    # SVM CLASSIFICATION
    # =====================================

    prediction = svm_model.predict(features)

    # =====================================
    # CONFIDENCE SCORE
    # =====================================

    confidence = 0.0

    try:

        probabilities = svm_model.predict_proba(features)

        confidence = np.max(probabilities) * 100

    except:

        confidence = 0.0

    # =====================================
    # GET CLASS NAME
    # =====================================

    class_name = classes[prediction[0]]

    return class_name, confidence

# =====================================
# TEST PREDICTION
# =====================================

if __name__ == "__main__":

    result, confidence = predict_image(
        "test_images/stroberi_1.jpg"
    )

    print("\n🍓 HASIL PREDIKSI")
    print("==========================")
    print("Kelas       :", result)
    print("Confidence  :", round(confidence, 2), "%")