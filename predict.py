import torch
import torch.nn as nn
import joblib
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# =====================================
# LOAD GATEKEEPER MODEL (Pretrained ImageNet, MobileNetV3 Small)
# =====================================

gatekeeper_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
gatekeeper_model = mobilenet_v3_small(weights=gatekeeper_weights)
gatekeeper_model.eval()

# Ambil daftar nama kelas ImageNet (1000 kelas)
imagenet_classes = gatekeeper_weights.meta["categories"]

# Transform khusus untuk gatekeeper (sesuai standar ImageNet)
gatekeeper_transform = gatekeeper_weights.transforms()

# Kata kunci yang dianggap valid sebagai "stroberi"
STRAWBERRY_KEYWORDS = ["strawberry"]

print("✅ Gatekeeper model (MobileNetV3 Small ImageNet) berhasil diload")


def is_strawberry(image_path, top_k=5):
    """
    Cek apakah gambar mengandung objek stroberi
    menggunakan model pretrained ImageNet.
    Return: (bool, top_predictions)
    """

    image = Image.open(image_path).convert("RGB")
    image_tensor = gatekeeper_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = gatekeeper_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Ambil top-k prediksi
    top_probs, top_idxs = torch.topk(probabilities, top_k)

    top_predictions = []
    found_strawberry = False

    for prob, idx in zip(top_probs, top_idxs):
        class_name = imagenet_classes[idx.item()]
        confidence = prob.item() * 100
        top_predictions.append((class_name, round(confidence, 2)))

        # Cek apakah nama kelas mengandung kata kunci stroberi
        if any(keyword in class_name.lower() for keyword in STRAWBERRY_KEYWORDS):
            found_strawberry = True

    return found_strawberry, top_predictions

# =====================================
# LOAD LABEL CLASS
# =====================================

with open("model/classes.txt", "r") as f:
    classes = [x.strip() for x in f.readlines()]

# =====================================
# LOAD CNN MODEL (MobileNetV3) - Model Utama
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

print("✅ MobileNetV3 (model utama) berhasil diload")

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

    # =====================================
    # STEP 1: GATEKEEPER CHECK
    # =====================================

    is_valid, top_preds = is_strawberry(image_path)

    print("📌 Top prediksi gatekeeper:", top_preds)

    if not is_valid:
        return "Bukan Stroberi", 0.0

    # =====================================
    # STEP 2: LANJUT KE CNN + SVM SEPERTI BIASA
    # =====================================

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