import argparse
import os
import torch
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
    parser = argparse.ArgumentParser(description="Training Config")
    parser.add_argument('--train_lowlight_dir', type=str, default="../Datasets/LOLv1/our485/low")
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
        
        # Using trans=True to resize images for training (prevents size mismatch in batches)
        self.train_loader = DataLoader(
            MyDataset(config.train_lowlight_dir, label_path=None, trans=True),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def train(self):
        print(f"Starting training for {self.config.epochs} epochs on {device}...")
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
            for batch_idx, (I_l, _, _, _) in enumerate(progress_bar):
                I_l = I_l.to(device)
                
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
                
                # Algorithm 1, Step 5: Calculate loss (Eq. 11)
                loss, stats = self.criterion(I_l, I_n, R_l_0, L_l_0, R_n_0, L_n_0, stage_outputs)
                
                # Backward pass
                loss.backward()
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

if __name__ == "__main__":
    config = parse_args()
    trainer = Trainer(config)
    trainer.train()
