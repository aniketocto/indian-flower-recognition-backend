from torchvision import datasets
import json

dataset = datasets.ImageFolder("data/all_flower")

with open("labels_aligned.json", encoding="utf-8") as f:
    labels = json.load(f)

for i, cls in enumerate(dataset.classes):
    assert labels[str(i)]["english"] == cls, f"Mismatch at {i}"

print("✅ labels.json perfectly aligned with training classes")
