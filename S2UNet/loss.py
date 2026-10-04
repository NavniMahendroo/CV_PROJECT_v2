import torch
import torch.nn as nn
import torch.nn.functional as F

class S2UNetLoss(nn.Module):
    def __init__(self, num_steps=3, lambda_1=0.15, lambda_2=0.05, lambda_3=0.1, theta=0.1, mu=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.theta = theta
        self.mu = mu

    def _get_max_rgb(self, I):
        # I shape: (B, 3, H, W)
        return torch.max(I, dim=1, keepdim=True)[0]

    def _gradient(self, img):
        # Compute horizontal and vertical gradients
        # img shape: (B, C, H, W)
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        
        # Pad to match original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        return grad_x, grad_y

    def _gradient_norm(self, img):
        grad_x, grad_y = self._gradient(img)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def _histogram_equalization(self, img):
        # Differentiable approximation is not needed since it's a target.
        # Compute empirical CDF and map values
        with torch.no_grad():
            B, C, H, W = img.size()
            target = torch.zeros_like(img)
            for b in range(B):
                for c in range(C):
                    img_flat = img[b, c].view(-1)
                    sorted_img, indices = torch.sort(img_flat)
                    cdf = torch.arange(1, H * W + 1, device=img.device, dtype=torch.float32) / (H * W)
                    
                    target_flat = torch.zeros_like(img_flat)
                    target_flat[indices] = cdf
                    target[b, c] = target_flat.view(H, W)
            return target

    def forward(self, I_l, I_n, R_l_0, L_l_0, R_n_0, L_n_0, stage_outputs):
        # 1. Initial Decomposer Loss (L_id)
        L_ir = F.l1_loss(R_l_0 * L_l_0, I_l) + F.l1_loss(R_n_0 * L_n_0, I_n)
        L_con = F.l1_loss(R_l_0, R_n_0)
        
        max_I_l = self._get_max_rgb(I_l)
        max_I_n = self._get_max_rgb(I_n)
        
        L_iie = F.l1_loss(L_l_0, max_I_l) + F.l1_loss(L_n_0, max_I_n)
        
        L_id = L_ir + L_con + L_iie

        # 2. Unfolding Module Loss (L_uf)
        L_rec = 0.0
        L_R = 0.0
        L_s = 0.0
        
        target_HE = self._histogram_equalization(max_I_l)
        
        for k in range(self.num_steps):
            R_k, L_l_k, L_n_k = stage_outputs[k]
            
            # L_rec
            L_rec += F.l1_loss(R_k * L_l_k, I_l) + F.l1_loss(R_k * L_n_k, I_n)
            
            # L_R
            omega_k = 1.0 if k == self.num_steps - 1 else 0.3
            max_R_k = self._get_max_rgb(R_k)
            
            term1 = self.lambda_1 * F.l1_loss(max_R_k, target_HE)
            term2 = self.lambda_2 * torch.mean(self._gradient_norm(R_k))
            L_R += omega_k * (term1 + term2)
            
            # L_s
            grad_R_k_norm = self._gradient_norm(R_k)
            weight_R = torch.exp(-self.theta * grad_R_k_norm)
            
            grad_L_l_k = self._gradient_norm(L_l_k)
            grad_L_n_k = self._gradient_norm(L_n_k)
            
            L_s += self.lambda_3 * (
                torch.mean(grad_L_l_k * weight_R) + 
                torch.mean(grad_L_n_k * weight_R)
            )

        # Final Illumination Loss (L_il)
        _, L_l_T, L_n_T = stage_outputs[-1]
        L_il = F.l1_loss(L_l_T, max_I_l) + F.l1_loss(L_n_T, max_I_n)
        
        L_uf = L_rec + L_R + L_s + self.mu * L_il
        
        L_total = L_id + L_uf
        
        # Compile stats dict for logging
        stats = {
            'L_total': L_total.item(),
            'L_id': L_id.item(),
            'L_uf': L_uf.item(),
            'L_rec': L_rec.item() if isinstance(L_rec, torch.Tensor) else L_rec,
            'L_R': L_R.item() if isinstance(L_R, torch.Tensor) else L_R,
            'L_s': L_s.item() if isinstance(L_s, torch.Tensor) else L_s
        }
        
        return L_total, stats
