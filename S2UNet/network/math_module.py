import torch
import torch.nn as nn
import torch.nn.functional as F


class P_Update(nn.Module):
    """
    p_k = R^{k-1} - λ1 * (L_l^{k-1} * (R^{k-1} * L_l^{k-1} - I_l) + L_n^{k-1} * (R^{k-1} * L_n^{k-1} - I_n))
    """

    def __init__(self, init_lambda=0.1):
        super().__init__()
        self.lambda_log = nn.Parameter(torch.log(torch.exp(torch.tensor(init_lambda)) - 1))

    @property
    def lambda_1(self):
        return F.softplus(self.lambda_log)

    def forward(self, R_prev, L_l_prev, I_l, L_n_prev, I_n):
        term1 = L_l_prev * (R_prev * L_l_prev - I_l)
        term2 = L_n_prev * (R_prev * L_n_prev - I_n)
        p_k = R_prev - self.lambda_1 * (term1 + term2)
        return p_k


class Q_Low_Update(nn.Module):
    """
    q_l^k = L_l^{k-1} - λ2 * R_k * (R_k * L_l^{k-1} - I_l)
    """

    def __init__(self, init_lambda=0.1):
        super().__init__()
        self.lambda_log = nn.Parameter(torch.log(torch.exp(torch.tensor(init_lambda)) - 1))

    @property
    def lambda_2(self):
        return F.softplus(self.lambda_log)

    def forward(self, L_l_prev, R_k, I_l):
        """
        L_l_prev : (B,1,H,W)
        R_k      : (B,3,H,W)
        I_l      : (B,3,H,W)
        """
        terms = R_k * (R_k * L_l_prev - I_l)  # (B,3,H,W)
        q_l_k = (L_l_prev - self.lambda_2 * terms).mean(dim=1, keepdim=True)

        return q_l_k


class Q_Norm_Update(nn.Module):
    """
    q_n^k = L_n^{k-1} - λ3 * R_k * (R_k * L_n^{k-1} - I_n)
    """

    def __init__(self, init_lambda=0.1):
        super().__init__()
        self.lambda_log = nn.Parameter(torch.log(torch.exp(torch.tensor(init_lambda)) - 1))

    @property
    def lambda_3(self):
        return F.softplus(self.lambda_log)

    def forward(self, L_n_prev, R_k, I_n):
        """
        L_n_prev : (B,1,H,W)
        R_k      : (B,3,H,W)
        I_n      : (B,3,H,W)
        """
        terms = R_k * (R_k * L_n_prev - I_n)  # (B,3,H,W)
        q_n_k = (L_n_prev - self.lambda_3 * terms).mean(dim=1, keepdim=True)

        return q_n_k

