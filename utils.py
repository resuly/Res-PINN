import os, pickle, json, copy, time, glob, gc
from pathlib import Path

import numpy as np
import pandas as pd

import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from quality_metrics import fsim, psnr
from joblib import Parallel, delayed, parallel_backend

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
        
class ParamsDynamic():

    def __init__(self, dict_params):
        self.__dict__.update(dict_params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


# Metrics
def rmse(predictions, targets):
    """Compute root mean squared error"""
    return np.sqrt(((predictions - targets) ** 2).mean())
def mse(predictions, targets):
    """Compute mean squared error"""
    return ((predictions - targets) ** 2).mean()
def mae(predictions, targets):
    """Compute mean absolute error"""
    return np.sum(np.absolute((predictions - targets))) / len(targets)
# def mape(predictions, targets):
#     """Compute mean absolute precentage error"""
#     mask = targets != 0
#     return (np.fabs(targets[mask] - predictions[mask])/targets[mask]).mean()
#     # return np.mean(np.absolute((targets - predictions) / targets)) * 100
def mape(a, b): 
    mask = a != 0
    return (np.fabs(a - b)[mask]/a[mask]).mean()


def get_next_version(root_dir):
    # get next version number 
    # Tensor Board root_dir/version_{}
    if not os.path.exists(root_dir):
        return 0
    
    existing_versions = []
    for d in os.listdir(root_dir):
        if d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))
            
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1

from torch.optim import Optimizer
# https://github.com/skyday123/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
# https://github.com/clovaai/AdamP/issues/5
class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (default: 0)
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962v5
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Debiasing
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                adam_step = exp_avg_hat / (exp_avg_sq_hat.sqrt().add(group['eps']))

                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                weight_norm = torch.norm(p.data)
                adam_norm = torch.norm(adam_step)
                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0

                p.data.add_(adam_step, alpha=-group['lr'] * trust_ratio)

        return loss


def show_st(df, mode='speed'):
    fig, ax = plt.subplots(1,1, figsize=(12,5))
    sns.heatmap(pd.pivot_table(df, values=f'{mode}', index='distance_raw', columns='time_raw'), cmap="jet_r", ax=ax)# vmin=0, vmax=50, 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    plt.show()