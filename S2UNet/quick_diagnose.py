import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from S2UNet.dataset import MyDataset
from S2UNet.network.total_module import RetinexUnfoldingNetUnsupervised
from S2UNet.loss import S2UNetLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_diagnostics(low_dir, gt_dir=None, batch_size=4):
    print("=" * 75)
    print("RUNNING DEEP DIAGNOSTIC ON DATA PIPELINE & TRAINING DYNAMICS")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Low-light Dir: {low_dir}")
    print(f"GT Dir:        {gt_dir}")

    # 1. Setup DataLoader
    if gt_dir is None:
        for candidate in [
            low_dir.replace('/low', '/high').replace('\\low', '\\high'),
            low_dir.replace('/low', '/normal').replace('\\low', '\\normal'),
        ]:
            if os.path.exists(candidate):
                gt_dir = candidate
                break

    dataset = MyDataset(low_dir, label_path=gt_dir, trans=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_data = next(iter(loader))

    if len(batch_data) == 4:
        I_l, I_gt, low_names, gt_names = batch_data
    else:
        I_l = batch_data[0]
        I_gt = None

    I_l = I_l.to(device)
    if isinstance(I_gt, torch.Tensor):
        I_gt = I_gt.to(device)

    # --------------------------------------------------------------------------
    # Check 1: Data loader value ranges
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 1: DATA LOADER VALUE RANGES =====")
    print(f"  Input Low-Light (I_l) : min={I_l.min().item():.4f}, max={I_l.max().item():.4f}, mean={I_l.mean().item():.4f}")
    if isinstance(I_gt, torch.Tensor):
        print(f"  Reference GT    (I_gt): min={I_gt.min().item():.4f}, max={I_gt.max().item():.4f}, mean={I_gt.mean().item():.4f}")
        if I_l.max().item() > 1.5 or I_gt.max().item() > 1.5:
            print("  [ERROR] Values are in [0, 255] range! Normalization is MISSING!")
        else:
            print("  [OK] Normalization confirmed: values are in [0, 1] range.")
    else:
        print("  [Notice] GT reference not loaded. Provide valid GT directory.")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Check 2: Input vs GT baseline PSNR
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 2: INPUT vs GT BASELINE PSNR =====")
    if isinstance(I_gt, torch.Tensor):
        mse_base = F.mse_loss(I_l, I_gt)
        input_psnr = (10.0 * torch.log10(1.0 / (mse_base + 1e-10))).item()
        print(f"  Baseline Input Low-Light PSNR vs GT: {input_psnr:.2f} dB")
        if 6.0 <= input_psnr <= 9.5:
            print("  [OK] Input PSNR is in expected range for LOL (~7-8 dB). Pairing & alignment verified.")
        else:
            print(f"  [WARNING] Input PSNR is {input_psnr:.2f} dB (expected ~7-8 dB)! Check data alignment, pairing, or resizing!")
    else:
        print("  [Notice] GT reference not loaded. Cannot compute baseline input PSNR.")
    print("=" * 70)

    # 2. Setup Model & Loss
    model = RetinexUnfoldingNetUnsupervised(num_steps=3).to(device)
    criterion = S2UNetLoss(num_steps=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # --------------------------------------------------------------------------
    # Check 5 (Part A): Initial learnable lambdas
    # --------------------------------------------------------------------------
    def get_lambdas():
        return [(i + 1, m['p'].lambda_1.item(), m['q_low'].lambda_2.item(), m['q_norm'].lambda_3.item())
                for i, m in enumerate(model.gradient_steps)]

    init_lambdas = get_lambdas()
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 5: LEARNABLE STEP SIZES LAMBDA (INITIAL) =====")
    for s, l1, l2, l3 in init_lambdas:
        print(f"  Stage {s}: lambda_p = {l1:.6f} | lambda_q_low = {l2:.6f} | lambda_q_norm = {l3:.6f}")
    print("=" * 70)

    # 3. Forward Pass
    gamma_map = model.gamma_predictor(I_l)
    I_n = torch.pow(I_l, gamma_map)
    s = 1000.0
    I_l_noisy = torch.poisson(I_l.clamp(min=1e-6, max=1.0) * s) / s
    I_n_noisy = torch.poisson(I_n.clamp(min=1e-6, max=1.0) * s) / s

    R_l_0, L_l_0, R_n_0, L_n_0, stage_outputs = model(
        I_l, I_n=I_n, I_l_noisy=I_l_noisy, I_n_noisy=I_n_noisy
    )
    R_T, L_l_T, L_n_T = stage_outputs[-1]

    # --------------------------------------------------------------------------
    # Check 3: Output range check & clamping
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 3: FORWARD PASS OUTPUT VALUE RANGES =====")
    print(f"  Initial Reflectance (R_l_0): min={R_l_0.min().item():.4f}, max={R_l_0.max().item():.4f}")
    print(f"  Initial Illum       (L_l_0): min={L_l_0.min().item():.4f}, max={L_l_0.max().item():.4f}")
    print(f"  Final Reflectance   (R_T)  : min={R_T.min().item():.4f}, max={R_T.max().item():.4f}")
    print(f"  Final Illum         (L_l_T): min={L_l_T.min().item():.4f}, max={L_l_T.max().item():.4f}")

    needs_clamp = (
        R_l_0.min() < 0.0 or R_l_0.max() > 1.0 or
        L_l_0.min() < 0.0 or L_l_0.max() > 1.0 or
        R_T.min() < 0.0 or R_T.max() > 1.0 or
        L_l_T.min() < 0.0 or L_l_T.max() > 1.0
    )
    if needs_clamp:
        print("  [ACTION] Output has values outside [0, 1]! Clamping to [0, 1] before loss.")
        R_l_0 = R_l_0.clamp(0.0, 1.0)
        L_l_0 = L_l_0.clamp(0.0, 1.0)
        R_n_0 = R_n_0.clamp(0.0, 1.0)
        L_n_0 = L_n_0.clamp(0.0, 1.0)
        stage_outputs = [(r.clamp(0.0, 1.0), ll.clamp(0.0, 1.0), ln.clamp(0.0, 1.0)) for (r, ll, ln) in stage_outputs]
    else:
        print("  [OK] Output values are already within [0, 1]. No clamping required.")
    print("=" * 70)

    # 4. Compute Loss
    loss, stats = criterion(I_l, I_n, R_l_0, L_l_0, R_n_0, L_n_0, stage_outputs)

    # --------------------------------------------------------------------------
    # Check 6: Individual loss terms balance check
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 6: INDIVIDUAL LOSS TERMS BALANCE =====")
    loss_breakdown = {
        'L_ir  (Init Recon)'       : stats.get('L_ir', 0.0),
        'L_con (Consistency)'      : stats.get('L_con', 0.0),
        'L_iie (Init Illum Est)'   : stats.get('L_iie', 0.0),
        'L_rec (Unfolding Recon)'  : stats.get('L_rec', 0.0),
        'L_R   (Reflectance Reg)'  : stats.get('L_R', 0.0),
        'L_s   (Smoothness)'       : stats.get('L_s', 0.0),
        'L_il  (Final Illum Loss)' : stats.get('L_il', 0.0),
    }
    vals = [v for v in loss_breakdown.values() if v > 0]
    avg_val = sum(vals) / len(vals) if vals else 1.0
    for term_name, val in loss_breakdown.items():
        flag = ""
        if val > 10.0 * avg_val:
            flag = " <-- [NOTE: >10x larger than average! Dominating the loss]"
        elif val < 0.1 * avg_val and val > 0:
            flag = " <-- [NOTE: >10x smaller than average]"
        print(f"  {term_name:25s}: {val:.6f}{flag}")
    print(f"  -------------------------------------------------")
    print(f"  Summed Total Loss        : {loss.item():.6f}")
    print("=" * 70)

    # 5. Backward Pass
    optimizer.zero_grad()
    loss.backward()

    # --------------------------------------------------------------------------
    # Check 4: Parameter gradient check
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 4: PARAMETER GRADIENTS CHECK (AFTER BACKWARD) =====")
    none_grad_count = 0
    total_trainable = 0
    none_grad_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += 1
            if param.grad is None:
                none_grad_count += 1
                none_grad_names.append(name)
    print(f"  Total Trainable Parameters   : {total_trainable}")
    print(f"  Parameters with grad == None : {none_grad_count}")
    if none_grad_count > 0:
        print(f"  [WARNING] {none_grad_count} parameters are NOT training (grad is None):")
        for n in none_grad_names[:6]:
            print(f"    - {n}")
        if len(none_grad_names) > 6:
            print(f"    ... and {len(none_grad_names) - 6} more.")
    else:
        print("  [OK] All trainable parameters received gradients successfully.")
    print("=" * 70)

    # 6. Step Optimizer & Check Lambda Delta
    optimizer.step()
    post_lambdas = get_lambdas()

    # --------------------------------------------------------------------------
    # Check 5 (Part B): Post-step lambda update verification
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("===== DIAGNOSTIC 5: STEP SIZES LAMBDA UPDATE CHECK (1 STEP) =====")
    any_changed = False
    for (st, l1_i, l2_i, l3_i), (_, l1_f, l2_f, l3_f) in zip(init_lambdas, post_lambdas):
        d1 = abs(l1_f - l1_i)
        d2 = abs(l2_f - l2_i)
        d3 = abs(l3_f - l3_i)
        print(f"  Stage {st} Delta: d(lambda_p)={d1:.6f}, d(lambda_q_low)={d2:.6f}, d(lambda_q_norm)={d3:.6f}")
        if d1 > 1e-6 or d2 > 1e-6 or d3 > 1e-6:
            any_changed = True

    if any_changed:
        print("  [OK] Learnable step sizes lambda successfully updated with optimizer step.")
    else:
        print("  [WARNING] Step sizes lambda DID NOT CHANGE! They are not properly updating as parameters.")
    print("=" * 70 + "\n")
    print("DIAGNOSTIC RUN COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lowlight_dir', type=str, default="../Datasets/LOLv1/our485/low")
    parser.add_argument('--train_gt_dir', type=str, default=None)
    args = parser.parse_args()

    run_diagnostics(args.train_lowlight_dir, args.train_gt_dir)
