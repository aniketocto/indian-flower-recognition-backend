from PIL import Image
from torchvision import transforms

from app.core.config import IMG_SIZE


# -----------------------
# Image Transformation (Validation-style)
# -----------------------

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image: Image.Image):
    """
    Convert PIL Image to model-ready tensor.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = _transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    return tensor