from torchvision import transforms
from dataset import Flower102Dataset

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = Flower102Dataset(
    image_dir = "data/jpg",
    labels_file = "data/imagelabels.mat",
    transform = transforms
)

print("Total images: ", len(dataset))
img, label = dataset[0]
print("Image shape: ", img.shape)
print("Label: ", label)