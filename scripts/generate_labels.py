import json
from torchvision import datasets

# 🔁 Use the SAME path used during training
DATASET_PATH = "data/all_flower"

# Load dataset to get class-to-index mapping
dataset = datasets.ImageFolder(DATASET_PATH)

labels = {}

for idx, class_name in enumerate(dataset.classes):
    labels[str(idx)] = {
        "english": class_name,
        "hindi": "",
        "marathi": "",
        "scientific": ""
    }

# Save labels.json
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)

print("✅ labels.json created successfully")
