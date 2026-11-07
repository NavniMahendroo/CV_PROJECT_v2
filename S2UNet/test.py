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
    parser.add_argument('--use_tta', action='store_true', help='Improvement 1: Enable 8-fold Test-Time Augmentation (Self-Ensemble)')

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

    def _forward_model(self, img):
        """Forward pass with optional 8-fold Test-Time Augmentation (Self-Ensemble)."""
        if not getattr(self.config, 'use_tta', False):
            return self.model(img)

        # Dihedral D4 group: 4 rotations x 2 flips
        transforms = [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
            (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
            (lambda x: torch.rot90(x, 1, dims=[2, 3]), lambda x: torch.rot90(x, -1, dims=[2, 3])),
            (lambda x: torch.rot90(x, 2, dims=[2, 3]), lambda x: torch.rot90(x, -2, dims=[2, 3])),
            (lambda x: torch.rot90(x, 3, dims=[2, 3]), lambda x: torch.rot90(x, -3, dims=[2, 3])),
            (lambda x: torch.rot90(torch.flip(x, dims=[2]), 1, dims=[2, 3]),
             lambda x: torch.flip(torch.rot90(x, -1, dims=[2, 3]), dims=[2])),
            (lambda x: torch.rot90(torch.flip(x, dims=[3]), 1, dims=[2, 3]),
             lambda x: torch.flip(torch.rot90(x, -1, dims=[2, 3]), dims=[3])),
        ]
        r_list, l_list = [], []
        for tf, inv_tf in transforms:
            aug_in = tf(img)
            r_aug, l_aug, _ = self.model(aug_in)
            r_list.append(inv_tf(r_aug))
            l_list.append(inv_tf(l_aug))

        r_ens = torch.mean(torch.stack(r_list, dim=0), dim=0)
        l_ens = torch.mean(torch.stack(l_list, dim=0), dim=0)
        return r_ens, l_ens, None

    def run(self):
        r_means = []
        l_means = []

        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing", leave=True)
            for low_imgs, _, low_names, _ in progress_bar:
                low_imgs = low_imgs.to(device)
                R, L_l, _ = self._forward_model(low_imgs)
                r_means.append(R.mean().item())
                l_means.append(L_l.mean().item())

                enhanced = R.clamp(0, 1)
                for i in range(enhanced.size(0)):
                    single_enhanced = enhanced[i].unsqueeze(0)
                    filename = os.path.basename(low_names[i])
                    self._save_image(single_enhanced, filename)

        # Print post-inference test set diagnostics
        mean_r = sum(r_means) / len(r_means) if r_means else 0.0
        mean_l = sum(l_means) / len(l_means) if l_means else 0.0

        print("\n" + "=" * 70)
        print("===== TEST SET DECOMPOSITION DIAGNOSTIC (FIXED GAMMA 0.8) =====")
        print(f"  Gamma (Fixed)            : 0.8000  [Original Paper Baseline]")
        if getattr(self.config, 'use_tta', False):
            print("  Test-Time Augmentation   : ACTIVE (8-fold Dihedral D4 Self-Ensemble) [Improvement 1]")
        else:
            print("  Test-Time Augmentation   : DISABLED (Standard single-pass inference)")
        print(f"  Reflectance Mean (R_T)   : {mean_r:.4f}  ", end="")
        if mean_r < 0.30:
            print("[WARNING: R is too dark! (< 0.30)]")
        else:
            print("[OK: R has healthy brightness (>= 0.30)]")

        print(f"  Illumination Mean (L_l_T): {mean_l:.4f}  ", end="")
        if mean_l > 0.70:
            print("[WARNING: L is too bright! (> 0.70)]")
        else:
            print("[OK: L is in expected range (<= 0.70)]")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    config = parse_args()
    tester = Tester(config)
    tester.run()
