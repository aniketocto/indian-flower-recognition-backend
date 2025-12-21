import os
import cv2
import shutil

# --------------------
# CONFIG
# --------------------
DATASET_DIR = "data/indian_flower"
BLUR_DIR = "data/indian_flower_blur_removed"
BLUR_THRESHOLD = 30  # SAFE threshold

os.makedirs(BLUR_DIR, exist_ok=True)

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

summary = {}

print("\nStarting blur check...\n")

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    blur_class_path = os.path.join(BLUR_DIR, class_name)
    os.makedirs(blur_class_path, exist_ok=True)

    total = 0
    removed = 0

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = variance_of_laplacian(gray)

        total += 1

        if blur_score < BLUR_THRESHOLD:
            shutil.move(img_path, os.path.join(blur_class_path, img_name))
            removed += 1

    summary[class_name] = (total, removed)

# --------------------
# REPORT
# --------------------
print("\nBLUR FILTER REPORT (SAFE MODE)")
print("--------------------------------")
for cls, (tot, rem) in summary.items():
    print(f"{cls}: {tot} images → {tot - rem} kept, {rem} moved")

print("\nDone. No files were deleted.")
