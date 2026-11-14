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
parser.add_argument('--test_dir', type=str, default='./result_test/LOL')
parser.add_argument('--test_gt_dir', type=str, default=None, help="Folder of GT images (optional).")
args = parser.parse_args()


def rgb_to_y_bt601(tensor):
    """Standard ITU-R BT.601 RGB to Y conversion (normalized [0, 1])."""
    return 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]


def rgb_to_y_matlab(tensor):
    """Matches MATLAB rgb2ycbcr Y-channel on [0, 1] inputs."""
    return (65.481 * tensor[:, 0:1, :, :] + 128.553 * tensor[:, 1:2, :, :] + 24.966 * tensor[:, 2:3, :, :] + 16.0) / 255.0


def evaluate_metric(in_dir, gt_dir=None):
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # ----- load input images (filter out non-image files) -----
    all_inputs = sorted(glob(os.path.join(in_dir, "*.*")))
    inputs = [f for f in all_inputs if f.lower().endswith(IMAGE_EXTENSIONS)]
    assert len(inputs) > 0, f"No input images found in {in_dir}."

    # ----- if GT exists, load and match it -----
    if gt_dir is not None and os.path.exists(gt_dir):
        all_gts = sorted(glob(os.path.join(gt_dir, "*.*")))
        gts = [f for f in all_gts if f.lower().endswith(IMAGE_EXTENSIONS)]
        if len(gts) != len(inputs):
            print(f"[Warning] GT count ({len(gts)}) != Input count ({len(inputs)}). PSNR/SSIM will not be computed.")
            gts = None
    else:
        gts = None

    has_gt = gts is not None

    rgb_psnrs, rgb_ssims = [], []
    y_psnrs, y_ssims = [], []
    matlab_y_psnrs, matlab_y_ssims = [], []
    niqes, pis, brisques = [], [], []
    per_image_results = []

    for idx, input_path in enumerate(tqdm(inputs, desc="Evaluating")):
        img_name = os.path.basename(input_path)
        input_img = Image.open(input_path).convert('RGB')
        input_tensor = T.ToTensor()(input_img).unsqueeze(0).to(device)

        # -------- Compute Full-Reference Metrics --------
        if has_gt:
            gt_img = Image.open(gts[idx]).convert('RGB')
            # Handle resolution difference if low and high are at different dimensions (e.g. LSRW)
            if gt_img.size != input_img.size:
                gt_img = gt_img.resize(input_img.size, Image.Resampling.BICUBIC)
            gt_tensor = T.ToTensor()(gt_img).unsqueeze(0).to(device)

            # 1. RGB Metrics
            rgb_psnr = psnr_metric(input_tensor, gt_tensor).item()
            rgb_ssim = ssim_metric(input_tensor, gt_tensor).item()
            rgb_psnrs.append(rgb_psnr)
            rgb_ssims.append(rgb_ssim)

            # 2. Y-channel (BT.601) Metrics
            in_y = rgb_to_y_bt601(input_tensor)
            gt_y = rgb_to_y_bt601(gt_tensor)
            y_psnr = psnr_metric(in_y, gt_y).item()
            y_ssim = ssim_metric(in_y, gt_y).item()
            y_psnrs.append(y_psnr)
            y_ssims.append(y_ssim)

            # 3. Y-channel (MATLAB formula) Metrics
            in_y_matlab = rgb_to_y_matlab(input_tensor)
            gt_y_matlab = rgb_to_y_matlab(gt_tensor)
            matlab_y_psnr = psnr_metric(in_y_matlab, gt_y_matlab).item()
            matlab_y_ssim = ssim_metric(in_y_matlab, gt_y_matlab).item()
            matlab_y_psnrs.append(matlab_y_psnr)
            matlab_y_ssims.append(matlab_y_ssim)

            per_image_results.append({
                'name': img_name,
                'rgb_psnr': rgb_psnr,
                'rgb_ssim': rgb_ssim,
                'y_psnr': y_psnr,
                'y_ssim': y_ssim,
                'matlab_y_psnr': matlab_y_psnr,
                'matlab_y_ssim': matlab_y_ssim
            })

        # -------- Compute No-Reference Metrics --------
        niqe = niqe_metric(input_tensor).item()
        pi = PI_metric(input_tensor).item()
        brisque = brisque_metric(input_tensor).item()

        niqes.append(niqe)
        pis.append(pi)
        brisques.append(brisque)

    result_dict = {
        'niqe': float(np.mean(niqes)),
        'pi': float(np.mean(pis)),
        'brisque': float(np.mean(brisques)),
        'has_gt': has_gt
    }

    if has_gt:
        result_dict.update({
            'rgb_psnr': float(np.mean(rgb_psnrs)),
            'rgb_ssim': float(np.mean(rgb_ssims)),
            'y_psnr': float(np.mean(y_psnrs)),
            'y_ssim': float(np.mean(y_ssims)),
            'matlab_y_psnr': float(np.mean(matlab_y_psnrs)),
            'matlab_y_ssim': float(np.mean(matlab_y_ssims)),
            'per_image': per_image_results
        })

    return result_dict


if __name__ == '__main__':
    # Init metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    niqe_metric = pyiqa.create_metric('niqe', device=device).eval()
    PI_metric = pyiqa.create_metric('pi', device=device).eval()
    brisque_metric = pyiqa.create_metric('brisque', device=device).eval()

    res = evaluate_metric(
        args.test_dir,
        args.test_gt_dir
    )

    print("\n" + "=" * 65)
    print("                     EVALUATION RESULTS")
    print("=" * 65)
    if res['has_gt']:
        print(f"  [RGB Color Space]        PSNR: {res['rgb_psnr']:.2f} dB  |  SSIM: {res['rgb_ssim']:.4f}")
        print(f"  [Y-Channel (BT.601)]     PSNR: {res['y_psnr']:.2f} dB  |  SSIM: {res['y_ssim']:.4f}")
        print(f"  [Y-Channel (MATLAB)]     PSNR: {res['matlab_y_psnr']:.2f} dB  |  SSIM: {res['matlab_y_ssim']:.4f}")
        print(f"  [No-Reference Metrics]   NIQE: {res['niqe']:.3f}  |  PI: {res['pi']:.3f}  |  BRISQUE: {res['brisque']:.2f}")
        print("-" * 65)
        print("  Paper Benchmark (LOL):   PSNR: 20.43 dB  |  SSIM: 0.7880")
        print("=" * 65)
        print("\nPer-Image Breakdown:")
        print(f"  {'Filename':<15} | {'RGB PSNR':<10} | {'Y PSNR':<10} | {'RGB SSIM':<10} | {'Y SSIM':<10}")
        print("  " + "-" * 62)
        for item in res['per_image']:
            print(f"  {item['name']:<15} | {item['rgb_psnr']:>6.2f} dB   | {item['y_psnr']:>6.2f} dB   | {item['rgb_ssim']:>8.4f}   | {item['y_ssim']:>8.4f}")
        print("=" * 65 + "\n")
    else:
        print("No GT provided.")
        print(f"NIQE:     {res['niqe']:.3f}")
        print(f"PI:       {res['pi']:.3f}")
        print(f"BRISQUE:  {res['brisque']:.2f}")
        print("=" * 65 + "\n")
