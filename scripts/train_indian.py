import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets

# ---------------
# Configuration
# ---------------
DATA_DIR = "data/indian_flower"
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 10
LEARNING_RATE = 0.003

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------
# Transforms
# ---------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
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

# ---------------
# Dataset and DataLoader
# ---------------
full_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=None  # Assign transforms after split
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

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
    shuffle=True,
    num_workers=0    
)

# ---------------
# Model
# ---------------
model = models.mobilenet_v2(weights=None)

# Replace classifier FIRST
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Load Oxford-trained weights EXCEPT classifier
state_dict = torch.load("flower102_mobilenet.pth", map_location=DEVICE)

# Remove classifier weights from Oxford checkpoint
state_dict.pop("classifier.1.weight")
state_dict.pop("classifier.1.bias")

model.load_state_dict(state_dict, strict=False)


#Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(DEVICE)


# ---------------
# Training Setup
# ---------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ---------------
# Training Loop
# ---------------
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

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

# --------------------
# Save model
# --------------------
torch.save(model.state_dict(), "indian_flower_mobilenet.pth")
print("Indian flower model saved.")
            