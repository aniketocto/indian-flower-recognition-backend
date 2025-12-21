import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
MODEL_PATH = "indian_flower_all_mobilenet.pth"
LABELS_PATH = "labels_aligned.json"
IMAGE_PATH = "data/test_set/20230923_175331.jpg"   # change for testing other images
OUTPUT_DIR = "outputs"
IMG_SIZE = 224
NUM_CLASSES = 37
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD LABELS =================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    idx_to_class = json.load(f)

# ================= LOAD MODEL =================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_pil = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

# ================= GRAD-CAM CORE =================
gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ================= FORWARD & BACKWARD =================
output = model(input_tensor)
pred_idx = output.argmax(dim=1).item()
label_info = idx_to_class[str(pred_idx)]

pred_english = label_info["english"]
pred_hindi = label_info["hindi"]
pred_marathi = label_info["marathi"]
pred_scientific = label_info["scientific"]

confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

model.zero_grad()
output[0, pred_idx].backward()

# ================= GENERATE HEATMAP =================
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * activations).sum(dim=1)
cam = F.relu(cam)
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam = cam[0].cpu().numpy()

# ================= OVERLAY =================
orig = cv2.imread(IMAGE_PATH)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
h, w, _ = orig.shape

cam_resized = cv2.resize(cam, (w, h))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

overlay = np.uint8(0.6 * orig + 0.4 * heatmap)

# ================= SAVE =================
output_path = os.path.join(
    OUTPUT_DIR,
    f"gradcam_{pred_english}.png"
)

print("Predicted:", pred_english)
Image.fromarray(overlay).save(output_path)

# ================= MATPLOTLIB DISPLAY =================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(orig)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title(
    f"{pred_english} ({confidence:.2%})\n"
    f"{pred_scientific}",
    fontsize=10
)
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Predicted:", pred_english)
print("Confidence:", confidence)
print("Saved at:", output_path)
