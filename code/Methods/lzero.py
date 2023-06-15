import numpy as np
import torch
from torch.distributions import Uniform


def hard(log_alpha: torch.Tensor, beta=2 / 3, gamma=-.1, xi=1.1, deterministic=False):
    if not deterministic:
        U = Uniform(0.0, 1.0)
        u = U.sample(log_alpha.shape).to(log_alpha.device)
    else:
        u = torch.tensor(0.5).to(log_alpha.device)
    log_epsilon = 1e-12
    s = torch.sigmoid((torch.log(u + log_epsilon) - torch.log(1 - u + log_epsilon) + log_alpha) / beta)
    s = s * (xi - gamma) + gamma
    return s.clamp(min=log_epsilon, max=1 - log_epsilon)


def structural_penalty(log_alpha, param, weight_decay, lambda_, mask_dim, beta=2 / 3, gamma=-0.1, xi=1.1):
    l2 = torch.sum(-(0.5 * weight_decay * param.pow(2)) - lambda_, mask_dim)
    l0 = torch.sigmoid(log_alpha - beta * np.log(-1 * gamma / xi))
    return torch.sum(l0 * l2)


def penalty(log_alpha, param, weight_decay, lambda_, beta=2 / 3, gamma=-0.1, xi=1.1):
    l0 = torch.sigmoid(log_alpha - beta * np.log(-1 * gamma / xi))
    return torch.sum(l0)
