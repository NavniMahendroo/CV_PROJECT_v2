import numpy as np
import os
import pyiqa
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("evaluate")
parser.add_argument('--test_dir', type=str, default='./result_test/LIME')
parser.add_argument('--test_gt_dir', type=str, default=None, help="Folder of GT images (optional).")
args = parser.parse_args()


def evaluate_metric(in_dir, gt_dir=None):

    # ----- load input images -----
    inputs = sorted(glob(os.path.join(in_dir, "*.*")))
    assert len(inputs) > 0, "No input images found."

    # ----- if GT exists, load and match it -----
    if gt_dir is not None and os.path.exists(gt_dir):
        gts = sorted(glob(os.path.join(gt_dir, "*.*")))
        if len(gts) != len(inputs):
            print("[Warning] GT count != Input count. PSNR/SSIM will not be computed.")
            gts = None
    else:
        gts = None  # No GT available

    has_gt = gts is not None

    psnrs, ssims = [], []
    niqes, pis, brisques = [], [], []

    for idx, input_path in enumerate(tqdm(inputs)):

        input_img = Image.open(input_path).convert('RGB')
        input_tensor = T.ToTensor()(input_img).unsqueeze(0).to(device)

        # -------- Compute Full-Reference Metrics --------
        if has_gt:
            gt_img = Image.open(gts[idx]).convert('RGB')
            gt_tensor = T.ToTensor()(gt_img).unsqueeze(0).to(device)

            psnr = psnr_metric(input_tensor, gt_tensor)
            ssim = ssim_metric(input_tensor, gt_tensor)

            psnrs.append(psnr.item())
            ssims.append(ssim.item())

        # -------- Compute No-Reference Metrics --------
        else:
            niqe = niqe_metric(input_tensor)
            pi = PI_metric(input_tensor)
            brisque = brisque_metric(input_tensor)

            niqes.append(niqe.item())
            pis.append(pi.item())
            brisques.append(brisque.item())

    if has_gt:
        ave_psnr = np.mean(psnrs)
        ave_ssim = np.mean(ssims)
        return ave_psnr, ave_ssim, None, None, None, True
    else:
        ave_niqe = np.mean(niqes)
        ave_pi = np.mean(pis)
        ave_brisque = np.mean(brisques)
        return None, None, ave_niqe, ave_pi, ave_brisque, False


if __name__ == '__main__':
    # Init metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    niqe_metric = pyiqa.create_metric('niqe', device=device).eval()
    PI_metric = pyiqa.create_metric('pi', device=device).eval()
    brisque_metric = pyiqa.create_metric('brisque', device=device).eval()

    ave_psnr, ave_ssim, ave_niqe, ave_pi, ave_brisque, has_gt = evaluate_metric(
        args.test_dir,
        args.test_gt_dir
    )

    print("\n===== Evaluation Results =====")
    if has_gt:
        print(f"PSNR:     {ave_psnr:.2f} dB")
        print(f"SSIM:     {ave_ssim:.4f}")
    else:
        print("No GT provided.")
        print(f"NIQE:     {ave_niqe:.3f}")
        print(f"PI:       {ave_pi:.3f}")
        print(f"BRISQUE:  {ave_brisque:.2f}")



