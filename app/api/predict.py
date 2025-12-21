from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import json

from app.core.model import load_model
from app.core.gradcam import generate_gradcam
from app.utils.image import preprocess_image
from app.core.config import LABELS_PATH, DEVICE

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Load image
    image = Image.open(file.file).convert("RGB")

    # Preprocess image
    input_tensor = preprocess_image(image).to(DEVICE)

    # Load model
    model = load_model()

    # Generate Grad-CAM
    pred_idx, confidence, gradcam_path = generate_gradcam(
        model, 
        input_tensor, 
        image
    )

    # Load labels
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    label_info = labels[str(pred_idx)]

    return {
        "prediction": {
            "english": label_info["english"],
            "hindi": label_info["hindi"],
            "marathi": label_info["marathi"],
            "scientific": label_info["scientific"],
            "confidence": round(confidence, 4)
        },
        "gradcam_image": str(gradcam_path) 
    }