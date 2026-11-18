# S2UNet: Low-Light Image Enhancement (v2)

> 💡 **Comprehensive Documentation & Benchmark Results:**  
> For the complete documentation, mathematical formulations, SOTA benchmark comparison tables, and version evolution (v1 vs. v2), please refer to the main repository [README.md](../README.md).

---

## Quick Reference Commands

### 1. Training (v2 with Cosine LR + Local Color Constancy Loss)
```bash
python train.py \
    --train_lowlight_dir /path/to/train/low \
    --train_gt_dir /path/to/train/high \
    --save_dir checkpoints_final \
    --epochs 120 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_cosine_lr \
    --lambda_color 0.05
```

### 2. Testing (v2 with Image-Level Adaptive Gamma)
```bash
python test.py \
    --test_lowlight_dir /path/to/test/low \
    --model_path checkpoints_final/checkpoint_epoch_120.pth \
    --save_dir result_test \
    --adaptive_gamma
```

### 3. Evaluation
```bash
# Full-reference (PSNR/SSIM across RGB, ITU-R BT.601, and MATLAB studio swing)
python evaluate.py --test_dir result_test --test_gt_dir /path/to/test/high

# Lightness Order Error (LOE)
python eval_loe.py --test_original_dir /path/to/test/low --test_processed_dir result_test
```
