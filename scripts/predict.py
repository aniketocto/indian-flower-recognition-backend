import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------
# Config
# --------------------
MODEL_PATH = "flower102_mobilenet.pth"
IMAGE_PATH = "data/test_set/Hibiscus_br_2_00633.jpg"
LABELS_PATH = "data/cat_to_name.json"
NUM_CLASSES = 102

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# Load label names
# --------------------
with open(LABELS_PATH, "r") as f:
    cat_to_name = json.load(f)

# Convert keys to int (JSON keys are strings)
cat_to_name = {int(k) - 1: v for k, v in cat_to_name.items()}

# --------------------
# Load model
# --------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --------------------
# Load image
# --------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# --------------------
# Inference
# --------------------
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)

top_probs, top_classes = torch.topk(probs, 3)

print("\nTop 3 Predictions:")
for i in range(3):
    class_idx = top_classes[0][i].item()
    flower_name = cat_to_name[class_idx]
    confidence = top_probs[0][i].item() * 100

    print(f"{i+1}. {flower_name} — {confidence:.2f}%")
