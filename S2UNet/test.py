import argparse
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from S2UNet.dataset import MyDataset
from S2UNet.network.total_module import RetinexUnfoldingNetUnsupervised_Test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Testing Config")

    parser.add_argument('--test_lowlight_dir', type=str, default="../Datasets/LOLv1/eval15/low")
    parser.add_argument('--model_path', type=str, default="./checkpoints/new_model_weights.pth")
    parser.add_argument('--save_dir', type=str, default="./result_test/LOL")

    return parser.parse_args()


class Tester:
    def __init__(self, config):
        self.config = config
        self.model = self._init_model()
        self.test_loader = self._init_dataloader()

        os.makedirs(config.save_dir, exist_ok=True)

    def _init_model(self):
        checkpoint = torch.load(
            self.config.model_path,
            map_location=device,
            weights_only=True
        )
        if 'num_steps' in checkpoint:
            step = checkpoint['num_steps']
        else:
            step = 3
        model = RetinexUnfoldingNetUnsupervised_Test(num_steps=step).to(device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _init_dataloader(self):
        return DataLoader(
            MyDataset(self.config.test_lowlight_dir, label_path=None, trans=False),
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _save_image(self, tensor, filename):
        tensor = tensor.squeeze(0).cpu().clamp(0, 1)
        numpy_img = (tensor.numpy() * 255.0).clip(0, 255).astype('uint8')
        if numpy_img.shape[0] == 3:
            numpy_img = numpy_img.transpose(1, 2, 0)
        output_img = Image.fromarray(numpy_img)
        output_img.save(os.path.join(self.config.save_dir, filename))

    def run(self):
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing", leave=True)
            for low_imgs, _, low_names, _ in progress_bar:
                low_imgs = low_imgs.to(device)
                R, _, _ = self.model(low_imgs)
                enhanced = R.clamp(0, 1)
                for i in range(enhanced.size(0)):
                    single_enhanced = enhanced[i].unsqueeze(0)
                    filename = os.path.basename(low_names[i])
                    self._save_image(single_enhanced, filename)


if __name__ == "__main__":
    config = parse_args()
    tester = Tester(config)
    tester.run()
