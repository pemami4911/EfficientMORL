import torch
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from scipy.stats import truncnorm


def truncated_normal_initializer(shape, mean, stddev):
    # compute threshold at 2 std devs
    values = truncnorm.rvs(mean - 2 * stddev, mean + 2 * stddev, size=shape)
    return torch.from_numpy(values).float()

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Modified from: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'truncated_normal':
                m.weight.data = truncated_normal_initializer(m.weight.shape, 0.0, stddev=init_gain)
            elif init_type == 'zeros':
                init.constant_(m.weight.data, 0.0)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def _softplus_to_std(softplus):
    softplus = torch.min(softplus, torch.ones_like(softplus)*80)
    return torch.sqrt(torch.log(1. + softplus.exp()) + 1e-5)

def mvn(loc, softplus, temperature=1.0):
    return torch.distributions.independent.Independent(
            torch.distributions.normal.Normal(loc, _softplus_to_std(softplus) * (1./temperature)), 1)

def std_mvn(shape, device):
    loc = torch.zeros(shape).to(device)
    scale = torch.ones(shape).to(device)
    return torch.distributions.independent.Independent(
            torch.distributions.normal.Normal(loc, scale), 1)


def gmm_negativeloglikelihood(x_t, x_loc, log_var, mask_logprobs):
    """
    mask_logprobs: [N, K, 1, H, W]
    """
    # NLL [batch_size, 1, H, W]
    sq_err = (x_t.unsqueeze(1) - x_loc).pow(2)
    # log N(x; x_loc, log_var): [N, K, C, H, W]
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    # [N, K, C, H, W]
    log_p_k = (mask_logprobs + normal_ll)
    # logsumexp over slots [N, C, H, W]
    log_p = torch.logsumexp(log_p_k, dim=1)
    # [N]
    nll = -torch.sum(log_p, dim=[1,2,3])
    return nll

def gaussian_negativeloglikelihood(x_t, x_loc, log_var):
    sq_err = (x_t - x_loc).pow(2)  # [N,C,H,W]
    # log N(x; x_loc, log_var): [N,C, H, W]
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    nll = -torch.sum(normal_ll, dim=[1,2,3])   # [N]
    return nll
