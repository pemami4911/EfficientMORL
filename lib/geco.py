import torch

class GECO:
    """
    see Taming VAEs (https://arxiv.org/abs/1810.00597) by Rezende, Viola
    """
    def __init__(self, nll_target, alpha=0.99, step_size_acceleration=1):
        """
        alpha: EMA decay
        """
        self.C = nll_target
        self.alpha = alpha
        self.step_size_acceleration = step_size_acceleration

    def step_beta(self, C_ema, beta, step_size):
        if C_ema > 0:
            step_size *= self.step_size_acceleration
        new_beta = beta.data - step_size * C_ema
        # softplus(0.55) = 1
        new_beta = torch.max(torch.tensor(0.55).to(beta.device), new_beta)
        beta.data = new_beta
        return beta

    def init_ema(self, C_ema, nll):
        C_ema.data = self.C - nll
        return C_ema

    def update_ema(self, C_ema, nll):
        C_ema.data = (C_ema.data * self.alpha).detach() + ((self.C - nll) * (1. - self.alpha))
        return C_ema

    def constraint(self, C_ema, beta, nll):
        # compute the constraint term
        C_t = (self.C - nll) 
        return torch.nn.functional.softplus(beta).detach() * (C_t + (C_ema - C_t).detach())
