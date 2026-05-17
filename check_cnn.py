import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

print("Load MobileNetV3...")

# Buat model dasar
model = mobilenet_v3_small()

# Ubah classifier menjadi 3 kelas
model.classifier[3] = nn.Linear(1024, 3)

# Load weight
state_dict = torch.load(
    "model/mobilenetv3_final.pth",
    map_location="cpu"
)

# Load model dengan strict=False
model.load_state_dict(
    state_dict,
    strict=False
)

model.eval()

print("MobileNetV3 berhasil diload")