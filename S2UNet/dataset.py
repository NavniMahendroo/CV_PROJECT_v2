import glob
import os
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils import data
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, train_path, label_path=None, trans=False, crop_size=256):
        self.imgs = sorted(glob.glob(os.path.join(train_path, "*.*")))
        if label_path is not None:
            self.labels = sorted(glob.glob(os.path.join(label_path, "*.*")))
        else:
            self.labels = None
            
        self.trans = trans
        self.crop_size = crop_size

    def _transform(self, img, label=None):
        if self.trans:
            # 1. Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            if label is not None:
                label = TF.crop(label, i, j, h, w)
            
            # 2. Random Horizontal Flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                if label is not None:
                    label = TF.hflip(label)
                    
            # 3. Random Vertical Flip
            if random.random() > 0.5:
                img = TF.vflip(img)
                if label is not None:
                    label = TF.vflip(label)

        # To Tensor
        img = TF.to_tensor(img)
        if label is not None:
            label = TF.to_tensor(label)
            return img, label
            
        return img

    def __getitem__(self, index):
        img_path = self.imgs[index]
        _, img_name = os.path.split(img_path)
        pil_img = Image.open(img_path).convert("RGB")

        if self.labels is not None:
            label_path = self.labels[index]
            _, label_name = os.path.split(label_path)
            pil_label = Image.open(label_path).convert("RGB")
            
            # Apply same transforms to both
            data_tensor, label_tensor = self._transform(pil_img, pil_label)
            return data_tensor, label_tensor, img_name, label_name
        else:
            # Apply transforms to input only
            data_tensor = self._transform(pil_img)
            return data_tensor, "", img_name, ""

    def __len__(self):
        return len(self.imgs)
