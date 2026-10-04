import argparse
import random

import numpy as np
import cv2
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("evaluate_LOE")
parser.add_argument('--test_original_dir', type=str, default='/mnt/data/24-LJ/Datasets/Unpair/LIME')
parser.add_argument('--test_processed_dir', type=str, default='./result_test/LIME')
args = parser.parse_args()


def calculate_loe(original, processed):
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8)
    if processed.dtype != np.uint8:
        processed = (processed * 255).astype(np.uint8)

    L_orig = np.max(original, axis=2).astype(np.float32)
    L_proc = np.max(processed, axis=2).astype(np.float32)

    h, w = L_orig.shape[:2]

    N = 100
    rows = np.round(np.linspace(0, h - 1, N)).astype(int)
    cols = np.round(np.linspace(0, w - 1, N)).astype(int)

    L_orig_small = L_orig[np.ix_(rows, cols)]
    L_proc_small = L_proc[np.ix_(rows, cols)]

    local_loe = np.zeros((N, N), dtype=np.int32)
    total_loe = 0

    for r in range(N):
        for c in range(N):
            orig_order = (L_orig_small >= L_orig_small[r, c])
            proc_order = (L_proc_small >= L_proc_small[r, c])

            order_error = np.logical_xor(orig_order, proc_order)

            current_errors = np.sum(order_error)
            local_loe[r, c] = current_errors
            total_loe += current_errors

    pixel_count = N * N
    total_loe = total_loe / pixel_count
    return total_loe, local_loe


def calculate_average_loe(original_dir, processed_dir):

    orig_files_map = {}
    for f in os.listdir(original_dir):
        base_name = os.path.splitext(f)[0]
        if base_name not in orig_files_map:
            orig_files_map[base_name] = f

    proc_files_map = {}
    for f in os.listdir(processed_dir):
        base_name = os.path.splitext(f)[0]
        if base_name not in proc_files_map:
            proc_files_map[base_name] = f

    common_base_names = set(orig_files_map.keys()) & set(proc_files_map.keys())

    if not common_base_names:
        raise ValueError("No matching image files found in both directories (based on filename base)")

    loe_scores = []
    print(f"Calculating LOE for {len(common_base_names)} images...")

    for base_name in tqdm(common_base_names):
        orig_filename = orig_files_map[base_name]
        proc_filename = proc_files_map[base_name]

        orig_path = os.path.join(original_dir, orig_filename)
        orig_img = cv2.imread(orig_path)
        if orig_img is None:
            print(f"Warning: Unable to read {orig_path}, skipping")
            continue

        proc_path = os.path.join(processed_dir, proc_filename)
        proc_img = cv2.imread(proc_path)
        if proc_img is None:
            print(f"Warning: Unable to read {proc_path}, skipping")
            continue

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

        if orig_img.shape[:2] != proc_img.shape[:2]:
            proc_img = cv2.resize(proc_img, (orig_img.shape[1], orig_img.shape[0]))

        try:
            total_loe, local_loe = calculate_loe(orig_img, proc_img)
            loe_scores.append(total_loe)

        except Exception as e:
            print(f"Error calculating LOE for {orig_filename} and {proc_filename}: {str(e)}")
            continue

    if not loe_scores:
        raise ValueError("All image calculations failed, unable to compute average")

    average_loe = np.mean(loe_scores)

    return average_loe


def set_seed(seed=5):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    average_loe = calculate_average_loe(
        args.test_original_dir,
        args.test_processed_dir
    )
    print("\n===== Evaluation Result =====")
    print(f"LOE: {average_loe:.2f}")
