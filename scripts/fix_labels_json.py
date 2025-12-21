from torchvision import datasets
import json

DATA_DIR = "data/all_flower"
OLD_JSON = "labels.json"
NEW_JSON = "labels_aligned.json"

# Load folder-based class order (used during training)
dataset = datasets.ImageFolder(DATA_DIR)
class_order = dataset.classes

with open(OLD_JSON, encoding="utf-8") as f:
    old_labels = json.load(f)

new_labels = {}

for idx, folder_name in enumerate(class_order):
    matched = False

    for _, info in old_labels.items():
        # Normalize names for matching
        json_name = info["english"].lower().replace("(", "").replace(")", "").replace("-", "").strip()
        folder_norm = folder_name.lower().replace("-", "").strip()

        if folder_norm in json_name or json_name in folder_norm:
            new_labels[str(idx)] = info
            matched = True
            break

    if not matched:
        raise ValueError(f"No JSON entry found for folder: {folder_name}")

with open(NEW_JSON, "w", encoding="utf-8") as f:
    json.dump(new_labels, f, ensure_ascii=False, indent=2)

print("✅ labels_aligned.json created successfully")
