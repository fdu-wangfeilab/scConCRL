

import scanpy as sc
import anndata as ad
import torch
import numpy as np

from scipy import sparse

from torch.distributions import Normal, kl_divergence, Laplace
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def plot_umap(adata, color=['celltype'], use_rep='X', save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3,3)) # type: ignore
    adata = adata.copy()
    print(adata)
    
    # pca make low dimension representation
    # sc.pp.scale(hvg_adata, max_value=10)
    # sc.tl.pca(adata)

    # using low dimension to calculate distance
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=color,wspace=0.4, legend_fontsize=14, show=None, save=save_filename)



def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"rmsprop": torch.optim.RMSprop, "adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]

    return optim_cls(params, **kwargs)


def squared_loss(output, target):
    n = output.shape[0]
    loss = 0.5 * torch.sum((output - target) ** 2) / n
    return loss


def kl_loss(mu, var, mu_p=None, var_p=None):
    var = torch.exp(0.5 * var.clamp(min=-10, max=10))
    if mu_p is None:
        kl_loss = kl_divergence(Normal(mu, var),
                            Normal(torch.zeros_like(mu), torch.ones_like(var)))
    else:
        var_p = torch.exp(0.5 * var_p.clamp(min=-10, max=10))
        kl_loss = kl_divergence(Normal(mu, var),
                            Normal(mu_p, var_p))
    
    if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
        print("Warning: Invalid values in KL loss")
        return torch.tensor(0.0).to(mu.device)
    
    return kl_loss.sum(dim=1).mean()

def kl_loss_lap(mu, log_scale, mu_p=None, log_scale_p=None):
    """
    计算两个拉普拉斯分布之间的 KL 散度。
    
    参数:
        mu: 拉普拉斯分布的均值 (tensor)
        log_scale: 拉普拉斯分布的对数尺度参数 (tensor)
        mu_p: 目标拉普拉斯分布的均值，默认为 0 (tensor, 可选)
        log_scale_p: 目标拉普拉斯分布的对数尺度参数，默认为 0 (tensor, 可选)
    
    返回:
        KL 散度的平均值 (scalar tensor)
    """
    # 将 log_scale 转换为尺度参数 scale = exp(log_scale)，并限制其范围以防数值溢出
    scale = torch.exp(log_scale.clamp(min=-10, max=10))
    
    if mu_p is None:
        # 默认目标分布为标准拉普拉斯分布 (mu=0, scale=1)
        target_dist = Laplace(torch.zeros_like(mu), torch.ones_like(scale))
    else:
        # 如果提供了目标分布的参数，则将其转换为目标分布
        scale_p = torch.exp(log_scale_p.clamp(min=-10, max=10))
        target_dist = Laplace(mu_p, scale_p)
    
    # 定义当前分布
    current_dist = Laplace(mu, scale)
    
    # 计算 KL 散度
    kl_loss = kl_divergence(current_dist, target_dist)
    
    # 检查是否有无效值 (NaN 或 Inf)
    if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
        print("Warning: Invalid values in KL loss")
        return torch.tensor(0.0).to(mu.device)
    
    # 返回 KL 散度的平均值
    return kl_loss.sum(dim=1).mean()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tc_loss(z_i, z_s):
    """
    计算Total Correlation (TC) loss
    :param z_i: 隐层元素 i (batch_size, latent_dim)
    :param z_s: 隐层元素 s (batch_size, latent_dim)
    :return: TC loss
    """
    batch_size, latent_dim = z_i.size()

    # 计算log q(z_i, z_s)
    log_qz_is = compute_log_density(torch.cat((z_i, z_s), dim=1))

    # 计算log q(z_i) + log q(z_s)
    log_qz_i = compute_log_density(z_i)
    log_qz_s = compute_log_density(z_s)

    # 计算TC loss
    tc = torch.sum(log_qz_is) - torch.sum(log_qz_i) - torch.sum(log_qz_s)
    return tc / batch_size

def compute_log_density(z):
    """
    计算log q(z)
    :param z: 输入变量 (batch_size, dim)
    :return: log density (batch_size,)
    """
    mean = torch.mean(z, dim=0)
    var = torch.var(z, dim=0)
    log_var = torch.log(var + 1e-8)
    log_density = -0.5 * torch.sum((z - mean) ** 2 / (var + 1e-8) + log_var + torch.log(torch.tensor(2 * torch.pi)), dim=1)
    return log_density


class EarlyStopping:
    """Cite from https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_model_dict(path) # save best model here
        self.val_loss_min = val_loss


def tensor2adata(x, val=1e-4):
    """
    Transform the tensor list to an Anndata.
    
    Parameters
    ----------
    x
        tensor list to concatenate.

    val
        value for sparse for matrix, default 1e-4.
        
    Returns
    ----------
    x
        concatenated anndata.
        
    """
    x = torch.cat(x)
    x = torch.masked_fill(x, x < val, 0)
    x = sparse.csr_matrix(x)
    x = ad.AnnData(x)
    return x

def get_denoised_adata(data, rep='X_denoised'):
    denoised_adata = sc.AnnData(X=data.obsm[rep])
    denoised_adata.obs = data.obs
    denoised_adata.var_names = data.var_names
    return denoised_adata


def tensor2numpy(x, val=1e-4, clip_data=True):
    """
    Transform the tensor list to an Anndata.
    
    Parameters
    ----------
    x
        tensor list to concatenate.

    val
        value for sparse for matrix, default 1e-4.
        
    Returns
    ----------
    x
        concatenated anndata.
        
    """
    if val is not None and clip_data:
        x = torch.masked_fill(x, x < val, 0)
    x = x.detach().cpu().numpy()
    return x





class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self.last_epoch = self.after_scheduler.last_epoch + self.total_epoch + 1
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'none':
            pass
        return loss