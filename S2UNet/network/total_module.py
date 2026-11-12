import torch
from torch import nn

from S2UNet.network.math_module import P_Update, Q_Low_Update, Q_Norm_Update
from S2UNet.network.illumination_low_prox import Q_Low_ProxNet
from S2UNet.network.init_decom import InitialDecomposer
from S2UNet.network.reflection_prox import P_ProxNet


class RetinexUnfoldingNetUnsupervised_Test(nn.Module):
    def __init__(self, num_steps: int = 3, init_lambda=0.2, gamma=0.8):
        super().__init__()
        self.num_steps = num_steps
        self.gamma = gamma

        self.decomposer = InitialDecomposer()

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

    def forward(self, I_l, gamma=None):
        # Support optional dynamic / adaptive gamma per image, with fallback to self.gamma (0.8)
        eff_gamma = gamma if gamma is not None else self.gamma
        I_n = torch.pow(I_l, eff_gamma)

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
    def __init__(self, num_steps: int = 3, init_lambda=0.2, gamma=0.8):
        super().__init__()
        self.num_steps = num_steps
        self.gamma = gamma

        self.decomposer = InitialDecomposer()

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

    def forward(self, I_l, I_n=None, I_l_noisy=None, I_n_noisy=None, return_diagnostics=False):
        if I_n is None:
            # Fixed gamma 0.8 (original paper formulation)
            I_n = torch.pow(I_l, self.gamma)

        # Use noisy versions for ID module decomposition if provided (training)
        # Use clean versions otherwise (inference)
        decomp_input_l = I_l_noisy if I_l_noisy is not None else I_l
        decomp_input_n = I_n_noisy if I_n_noisy is not None else I_n

        R_l, L_l = self.decomposer(decomp_input_l)
        R_n, L_n = self.decomposer(decomp_input_n)

        R_k, L_l_k, L_n_k = R_l, L_l, L_n
        
        stage_outputs = []
        stage_diagnostics = []

        for step in range(self.num_steps):
            modules = self.gradient_steps[step]
            
            # Record before values if diagnosing
            if return_diagnostics:
                r_pre = R_k.mean().item()
                l_pre = L_l_k.mean().item()

            # UF module uses CLEAN I_l, I_n for gradient computation (Eq. 7)
            # 1. Gradient descent step on R (p_k)
            q_k = modules['p'](R_k, L_l_k, I_l, L_n_k, I_n)
            # 2. Proximal operator on R
            R_k = modules['prox_r'](q_k)

            # 3. Gradient descent step on L_l and L_n (q_k)
            q_l_k = modules['q_low'](L_l_k, R_k, I_l)
            q_n_k = modules['q_norm'](L_n_k, R_k, I_n)
            # 4. Proximal operator on L_l and L_n
            L_l_k = modules['prox_l_l'](q_l_k)
            L_n_k = modules['prox_l_n'](q_n_k)
            
            stage_outputs.append((R_k, L_l_k, L_n_k))

            if return_diagnostics:
                stage_diagnostics.append({
                    'stage': step + 1,
                    'lambda_p': modules['p'].lambda_1.item(),
                    'lambda_q_low': modules['q_low'].lambda_2.item(),
                    'r_before': r_pre,
                    'r_after_grad': q_k.mean().item(),
                    'r_after_prox': R_k.mean().item(),
                    'l_before': l_pre,
                    'l_after_grad': q_l_k.mean().item(),
                    'l_after_prox': L_l_k.mean().item(),
                })

        if return_diagnostics:
            return R_l, L_l, R_n, L_n, stage_outputs, stage_diagnostics
        return R_l, L_l, R_n, L_n, stage_outputs
