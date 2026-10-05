import torch

from torch import nn

from S2UNet.network.math_module import P_Update, Q_Low_Update, Q_Norm_Update
from S2UNet.network.illumination_low_prox import Q_Low_ProxNet
from S2UNet.network.init_decom import InitialDecomposer
from S2UNet.network.reflection_prox import P_ProxNet


class SpatialGammaPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Branch 1: Understands the GLOBAL brightness of the entire image
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Understands the LOCAL details
        self.local_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Combines Global Brightness + Local Details to predict the final Gamma Map
        self.combiner = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 1. Measure overall image brightness
        global_feat = self.global_pool(x).view(B, C)
        global_feat = self.global_net(global_feat) # Shape: (B, 16)
        
        # Expand global knowledge to the whole image shape
        global_feat = global_feat.view(B, 16, 1, 1).expand(B, 16, H, W)
        
        # 2. Measure local features
        local_feat = self.local_net(x) # Shape: (B, 16, H, W)
        
        # 3. Combine them so Gamma knows BOTH local textures and global brightness
        concat_feat = torch.cat([local_feat, global_feat], dim=1)
        
        # 4. Predict adaptive gamma (allowing a wider dynamic range [0.3, 1.0])
        gamma_map = self.combiner(concat_feat) * 0.7 + 0.3
        return gamma_map


class RetinexUnfoldingNetUnsupervised_Test(nn.Module):
    def __init__(self, num_steps: int = 3, init_lambda=0.2):
        super().__init__()
        self.num_steps = num_steps

        self.decomposer = InitialDecomposer()
        self.gamma_predictor = SpatialGammaPredictor()

        self.gradient_steps = nn.ModuleList([
            nn.ModuleDict({
                'p': P_Update(init_lambda),
                'q_low': Q_Low_Update(init_lambda),
                'q_norm': Q_Norm_Update(init_lambda),
                'prox_r': P_ProxNet(),
                'prox_l_l': Q_Low_ProxNet(),
                'prox_l_n': Q_Low_ProxNet(),
            }) for i in range(num_steps)
        ])

    def forward(self, I_l):
        gamma_map = self.gamma_predictor(I_l)
        I_n = torch.pow(I_l, gamma_map)

        R_l, L_l = self.decomposer(I_l)
        R_n, L_n = self.decomposer(I_n)

        R_k, L_l_k, L_n_k = R_l, L_l, L_n

        for step in range(self.num_steps):
            modules = self.gradient_steps[step]
            q_k = modules['p'](R_k, L_l_k, I_l, L_n_k, I_n)
            R_k = modules['prox_r'](q_k)
            q_l_k = modules['q_low'](L_l_k, R_k, I_l)
            q_n_k = modules['q_norm'](L_n_k, R_k, I_n)
            L_l_k = modules['prox_l_l'](q_l_k)
            L_n_k = modules['prox_l_n'](q_n_k)

        return R_k, L_l_k, L_n_k


class RetinexUnfoldingNetUnsupervised(nn.Module):
    def __init__(self, num_steps: int = 3, init_lambda=0.2):
        super().__init__()
        self.num_steps = num_steps

        self.decomposer = InitialDecomposer()
        self.gamma_predictor = SpatialGammaPredictor()

        self.gradient_steps = nn.ModuleList([
            nn.ModuleDict({
                'p': P_Update(init_lambda),
                'q_low': Q_Low_Update(init_lambda),
                'q_norm': Q_Norm_Update(init_lambda),
                'prox_r': P_ProxNet(),
                'prox_l_l': Q_Low_ProxNet(),
                'prox_l_n': Q_Low_ProxNet(),
            }) for i in range(num_steps)
        ])

    def forward(self, I_l, I_n=None, I_l_noisy=None, I_n_noisy=None):
        if I_n is None:
            gamma_map = self.gamma_predictor(I_l)
            I_n = torch.pow(I_l, gamma_map)

        # Use noisy versions for ID module decomposition if provided (training)
        # Use clean versions otherwise (inference)
        decomp_input_l = I_l_noisy if I_l_noisy is not None else I_l
        decomp_input_n = I_n_noisy if I_n_noisy is not None else I_n

        R_l, L_l = self.decomposer(decomp_input_l)
        R_n, L_n = self.decomposer(decomp_input_n)

        R_k, L_l_k, L_n_k = R_l, L_l, L_n
        
        stage_outputs = []

        for step in range(self.num_steps):
            modules = self.gradient_steps[step]
            # UF module uses CLEAN I_l, I_n for gradient computation (Eq. 7)
            q_k = modules['p'](R_k, L_l_k, I_l, L_n_k, I_n)
            R_k = modules['prox_r'](q_k)
            q_l_k = modules['q_low'](L_l_k, R_k, I_l)
            q_n_k = modules['q_norm'](L_n_k, R_k, I_n)
            L_l_k = modules['prox_l_l'](q_l_k)
            L_n_k = modules['prox_l_n'](q_n_k)
            
            stage_outputs.append((R_k, L_l_k, L_n_k))

        return R_l, L_l, R_n, L_n, stage_outputs
