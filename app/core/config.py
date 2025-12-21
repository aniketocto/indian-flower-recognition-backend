from pathlib import Path

# ---------------------
# Base Directory
# ---------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------
# Paths
# --------------------- 
MODEL_PATH = BASE_DIR / "models" / "indian_flower_all_mobilenet.pth"
LABELS_PATH = BASE_DIR / "metadata" / "labels_aligned.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "gradcam"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------
# Image and Model Configurations
# ---------------------
NUM_CLASSES = 37
IMG_SIZE = 224

# ---------------------
# Runtime
# ---------------------
DEVICE = "cuda" 