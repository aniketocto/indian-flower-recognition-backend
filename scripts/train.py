import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from dataset import Flower102Dataset

# ---------------
# Configuration
# ---------------
BATCH_SIZE = 16
EPOCHS = 3
NUM_CLASSES = 102
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------
# Transforms training with data augmentation
# ---------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# validation transforms
val_transforms = transforms.Compose([
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
# create dataset without transforms first
full_dataset = Flower102Dataset(
    image_dir="data/jpg",
    labels_file="data/imagelabels.mat",
    transform=None
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Assign transforms AFTER the split
train_ds.dataset.transform = train_transforms
val_ds.dataset.transform = val_transforms

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
model = models.mobilenet_v2(weights = "IMAGENET1K_V1")

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ---------------
# Training Setup
# ---------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)


    #validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")


# ---------------
# Save Model
# ---------------
torch.save(model.state_dict(), "flower102_mobilenet.pth")
print("Model saved ")