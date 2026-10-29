import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from S2UNet.dataset import MyDataset
from S2UNet.network.total_module import RetinexUnfoldingNetUnsupervised
from S2UNet.loss import S2UNetLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Training Config with Diagnostic Checks")
    parser.add_argument('--train_lowlight_dir', type=str, default="../Datasets/LOLv1/our485/low")
    parser.add_argument('--train_gt_dir', type=str, default=None, help="Directory of GT reference images (e.g. ../Datasets/LOLv1/our485/high)")
    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--noise_scale', type=float, default=1000.0, help='Poisson noise scale s (Eq. 9)')
    return parser.parse_args()


class Trainer:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.save_dir, exist_ok=True)
        
        self.model = RetinexUnfoldingNetUnsupervised(
            num_steps=config.num_steps
        ).to(device)
        
        self.criterion = S2UNetLoss(num_steps=config.num_steps).to(device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        
        # Auto-detect GT reference directory if not provided
        gt_dir = config.train_gt_dir
        if gt_dir is None:
            for candidate in [
                config.train_lowlight_dir.replace('/low', '/high').replace('\\low', '\\high'),
                config.train_lowlight_dir.replace('/low', '/normal').replace('\\low', '\\normal'),
            ]:
                if os.path.exists(candidate):
                    gt_dir = candidate
                    break

        self.gt_dir = gt_dir
        # Using trans=True to resize images for training (prevents size mismatch in batches)
        self.train_loader = DataLoader(
            MyDataset(config.train_lowlight_dir, label_path=gt_dir, trans=True),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def get_lambda_values(self):
        """Helper for Diagnostic 5: Retrieves learnable step sizes lambda from unfolding module."""
        lambdas = []
        for step_idx, modules in enumerate(self.model.gradient_steps):
            l1 = modules['p'].lambda_1.item()
            l2 = modules['q_low'].lambda_2.item()
            l3 = modules['q_norm'].lambda_3.item()
            lambdas.append((step_idx + 1, l1, l2, l3))
        return lambdas

    def print_lambdas(self, title, lambdas):
        """Helper for Diagnostic 5: Prints step sizes lambda."""
        print("\n" + "=" * 70)
        print(f"===== DIAGNOSTIC 5: LEARNABLE STEP SIZES LAMBDA ({title}) =====")
        for stage, l1, l2, l3 in lambdas:
            print(f"  Stage {stage}: lambda_p = {l1:.6f} | lambda_q_low = {l2:.6f} | lambda_q_norm = {l3:.6f}")
        print("=" * 70)

    def train(self):
        print(f"Starting training for {self.config.epochs} epochs on {device}...")
        
        # Diagnostic 5: Record learnable lambdas at start of training
        initial_lambdas = self.get_lambda_values()
        self.print_lambdas("START OF TRAINING", initial_lambdas)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
            for batch_idx, batch_data in enumerate(progress_bar):
                # Unpack batch data
                if len(batch_data) == 4:
                    I_l, I_gt, _, _ = batch_data
                else:
                    I_l = batch_data[0]
                    I_gt = None

                I_l = I_l.to(device)
                if isinstance(I_gt, torch.Tensor):
                    I_gt = I_gt.to(device)

                # =============================================================
                # Diagnostic 1: Data loader value ranges check (Batch 0, Epoch 1)
                # =============================================================
                if batch_idx == 0 and epoch == 1:
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
                        print("  [Notice] GT reference not loaded. Pass --train_gt_dir to inspect reference tensor.")
                    print("=" * 70)

                # =============================================================
                # Diagnostic 2: Input vs GT baseline PSNR check (Batch 0, Epoch 1)
                # =============================================================
                if batch_idx == 0 and epoch == 1:
                    print("\n" + "=" * 70)
                    print("===== DIAGNOSTIC 2: INPUT vs GT BASELINE PSNR =====")
                    if isinstance(I_gt, torch.Tensor):
                        mse_base = F.mse_loss(I_l, I_gt)
                        input_psnr = (10.0 * torch.log10(1.0 / (mse_base + 1e-10))).item()
                        print(f"  Baseline Input Low-Light PSNR vs GT: {input_psnr:.2f} dB")
                        if 6.0 <= input_psnr <= 9.5:
                            print("  [OK] Input PSNR is in expected range for LOL (~7-8 dB). Data pairing and alignment verified.")
                        else:
                            print(f"  [WARNING] Input PSNR ({input_psnr:.2f} dB) is outside expected ~7-8 dB range! Check data alignment, pairing, or resizing!")
                    else:
                        print("  [Notice] GT reference not loaded. Pass --train_gt_dir to compute baseline input PSNR.")
                    print("=" * 70)

                self.optimizer.zero_grad()
                
                # Algorithm 1, Step 1: Predict Spatial Gamma Map
                gamma_map = self.model.gamma_predictor(I_l)
                I_n = torch.pow(I_l, gamma_map)
                
                # Algorithm 1, Step 2: Poisson noise injection (Eq. 9)
                s = self.config.noise_scale
                I_l_noisy = torch.poisson(I_l.clamp(min=1e-6, max=1.0) * s) / s
                I_n_noisy = torch.poisson(I_n.clamp(min=1e-6, max=1.0) * s) / s
                
                # Algorithm 1, Steps 3-4: Forward pass
                # ID module receives noisy inputs; UF module uses clean I_l, I_n
                R_l_0, L_l_0, R_n_0, L_n_0, stage_outputs = self.model(
                    I_l, I_n=I_n, I_l_noisy=I_l_noisy, I_n_noisy=I_n_noisy
                )

                # =============================================================
                # Diagnostic 3: Output range check & clamping (Batch 0, Epoch 1)
                # =============================================================
                R_T, L_l_T, L_n_T = stage_outputs[-1]
                if batch_idx == 0 and epoch == 1:
                    print("\n" + "=" * 70)
                    print("===== DIAGNOSTIC 3: FORWARD PASS OUTPUT VALUE RANGES (BEFORE LOSS) =====")
                    print(f"  Initial Reflectance (R_l_0): min={R_l_0.min().item():.4f}, max={R_l_0.max().item():.4f}")
                    print(f"  Initial Illum       (L_l_0): min={L_l_0.min().item():.4f}, max={L_l_0.max().item():.4f}")
                    print(f"  Final Reflectance   (R_T)  : min={R_T.min().item():.4f}, max={R_T.max().item():.4f}")
                    print(f"  Final Illum         (L_l_T): min={L_l_T.min().item():.4f}, max={L_l_T.max().item():.4f}")

                # Check if clamping to [0, 1] is needed
                needs_clamp = (
                    R_l_0.min() < 0.0 or R_l_0.max() > 1.0 or
                    L_l_0.min() < 0.0 or L_l_0.max() > 1.0 or
                    R_n_0.min() < 0.0 or R_n_0.max() > 1.0 or
                    L_n_0.min() < 0.0 or L_n_0.max() > 1.0 or
                    R_T.min() < 0.0 or R_T.max() > 1.0 or
                    L_l_T.min() < 0.0 or L_l_T.max() > 1.0
                )

                if needs_clamp:
                    if batch_idx == 0 and epoch == 1:
                        print("  [ACTION] Detected values outside [0, 1]! Clamping all outputs to [0, 1] before computing loss.")
                    R_l_0 = R_l_0.clamp(0.0, 1.0)
                    L_l_0 = L_l_0.clamp(0.0, 1.0)
                    R_n_0 = R_n_0.clamp(0.0, 1.0)
                    L_n_0 = L_n_0.clamp(0.0, 1.0)
                    clamped_stages = []
                    for (r_k, l_l_k, l_n_k) in stage_outputs:
                        clamped_stages.append((
                            r_k.clamp(0.0, 1.0),
                            l_l_k.clamp(0.0, 1.0),
                            l_n_k.clamp(0.0, 1.0)
                        ))
                    stage_outputs = clamped_stages
                else:
                    if batch_idx == 0 and epoch == 1:
                        print("  [OK] Output values are already within [0, 1]. No clamping required.")
                if batch_idx == 0 and epoch == 1:
                    print("=" * 70)
                
                # Algorithm 1, Step 5: Calculate loss (Eq. 11)
                loss, stats = self.criterion(I_l, I_n, R_l_0, L_l_0, R_n_0, L_n_0, stage_outputs)

                # =============================================================
                # Diagnostic 6: Individual loss terms balance check (Batch 0, Epoch 1)
                # =============================================================
                if batch_idx == 0 and epoch == 1:
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
                
                # Backward pass
                loss.backward()

                # =============================================================
                # Diagnostic 4: Parameter gradient check (Batch 0, Epoch 1)
                # =============================================================
                if batch_idx == 0 and epoch == 1:
                    print("\n" + "=" * 70)
                    print("===== DIAGNOSTIC 4: PARAMETER GRADIENTS CHECK (AFTER BACKWARD) =====")
                    none_grad_count = 0
                    total_trainable = 0
                    none_grad_names = []
                    for name, param in self.model.named_parameters():
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
                    print("=" * 70 + "\n")

                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            self.scheduler.step()
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch}/{self.config.epochs}] Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % 10 == 0 or epoch == self.config.epochs:
                save_path = os.path.join(self.config.save_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'num_steps': self.config.num_steps
                }, save_path)
                print(f"Checkpoint saved to {save_path}")

        # =====================================================================
        # Diagnostic 5: Check learnable step sizes lambda at end of training
        # =====================================================================
        final_lambdas = self.get_lambda_values()
        self.print_lambdas("END OF TRAINING", final_lambdas)

        print("\n" + "=" * 70)
        print("===== DIAGNOSTIC 5 VERIFICATION: DID STEP SIZES LAMBDA CHANGE? =====")
        any_changed = False
        for (st, l1_i, l2_i, l3_i), (_, l1_f, l2_f, l3_f) in zip(initial_lambdas, final_lambdas):
            d1 = abs(l1_f - l1_i)
            d2 = abs(l2_f - l2_i)
            d3 = abs(l3_f - l3_i)
            print(f"  Stage {st} Delta: d(lambda_p)={d1:.6f}, d(lambda_q_low)={d2:.6f}, d(lambda_q_norm)={d3:.6f}")
            if d1 > 1e-6 or d2 > 1e-6 or d3 > 1e-6:
                any_changed = True

        if any_changed:
            print("  [OK] Learnable step sizes lambda successfully updated during training.")
        else:
            print("  [WARNING] Step sizes lambda DID NOT CHANGE! They may not be properly registered as parameters.")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    config = parse_args()
    trainer = Trainer(config)
    trainer.train()
