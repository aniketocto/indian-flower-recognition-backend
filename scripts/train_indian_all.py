import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# --------------------
# CONFIG
# --------------------
DATA_DIR = "data/all_flower"

BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 37
LEARNING_RATE = 0.0003

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------
# Dataset
# --------------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Assign transforms AFTER split
train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------
# Model (expand from Phase-2 Indian model)
# --------------------
model = models.mobilenet_v2(weights=None)

# Replace classifier FIRST (10 → 37 classes)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Load Phase-2 Indian model weights (EXCEPT classifier)
state_dict = torch.load("indian_flower_mobilenet.pth", map_location=DEVICE)

# Remove old 10-class classifier weights
state_dict.pop("classifier.1.weight")
state_dict.pop("classifier.1.bias")

# Load remaining weights
model.load_state_dict(state_dict, strict=False)


# Freeze backbone to protect learned features
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(DEVICE)


# --------------------
# Training setup
# --------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

# --------------------
# Save model
# --------------------
torch.save(model.state_dict(), "indian_flower_all_mobilenet.pth")
print("Phase-3 Indian flower model saved.")
