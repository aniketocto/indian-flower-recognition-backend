import torch
import torch.nn as nn
from torchvision import models

from app.core.config import MODEL_PATH, NUM_CLASSES, DEVICE

# -----------------------
# Global Model Variable
# -----------------------
_model = None


def load_model():
    """
    Load the model only once (singleton pattern).
    """
    global _model
    if _model is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)

        _model = model
    return _model