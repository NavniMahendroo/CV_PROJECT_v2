import glob
import os

from torch.utils import data
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256), Image.Resampling.LANCZOS),
    transforms.ToTensor()
])


class MyDataset(data.Dataset):
    def __init__(self, train_path, label_path, trans=False):
        self.imgs = glob.glob(os.path.join(train_path, "*.*"))
        if label_path is not None:
            self.labels = glob.glob(os.path.join(label_path, "*.*"))
        else:
            self.labels = None
        if trans:
            self.transforms = transform
        else:
            self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        img = self.imgs[index]
        _, img_name = os.path.split(img)
        pil_img = Image.open(img)
        pil_img = pil_img.convert("RGB")
        data = self.transforms(pil_img)

        if self.labels is not None:
            label = self.labels[index]
            _, label_name = os.path.split(label)
            pil_label = Image.open(label)
            pil_label = pil_label.convert("RGB")
            label = self.transforms(pil_label)
            return data, label, img_name, label_name
        return data, "", img_name, ""

    def __len__(self):
        return len(self.imgs)
