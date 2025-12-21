import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import os

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "indian_flower_all_mobilenet.pth"
DATA_DIR = "data/all_flower"

# Put path of ONE test image here
IMAGE_PATH = "data/test_set/BrightEyes_2_00537.jpg"

NUM_CLASSES = 37
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------
# Image transforms (validation-style)
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
# Load class names from dataset folders
# --------------------
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes


# --------------------
# Load model
# --------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

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

top_probs, top_indices = torch.topk(probs, 5)

print("\nTop Predictions:")
for i in range(5):
    idx = top_indices[0][i].item()
    confidence = top_probs[0][i].item() * 100
    print(f"{i+1}. {class_names[idx]} — {confidence:.2f}%")
