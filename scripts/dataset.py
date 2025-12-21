import os
from PIL import Image
import scipy.io
from torch.utils.data import Dataset

class Flower102Dataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        mat = scipy.io.loadmat(labels_file)
        self.labels = mat['labels'][0] # 1-indexes labels

        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx] - 1  # Convert to 0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label