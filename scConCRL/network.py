import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from scConCRL.utils import FocalLossV1

class InvertibleMLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(InvertibleMLP, self).__init__()
        self.input = nn.Sequential(
                    nn.Linear(n_inputs, hparams.mlp_width),
                    nn.LeakyReLU(hparams.mlp_ac),
                )
        self.hiddens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hparams.mlp_width, hparams.mlp_width),
                    nn.LeakyReLU(hparams.mlp_ac),
                )
                for _ in range(hparams.mlp_depth - 2)
            ]
        )
        self.output = nn.Linear(hparams.mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        for hidden in self.hiddens:
            x = hidden(x)
        x = self.output(x)
        return x


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        
        super(MLP, self).__init__()
        self.input = nn.Sequential(
                    nn.Linear(n_inputs, hparams.mlp_width),
                    nn.LayerNorm(hparams.mlp_width),
                    nn.LeakyReLU(hparams.mlp_ac),
                    nn.Dropout(hparams.mlp_dropout)
                )
        self.hiddens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hparams.mlp_width, hparams.mlp_width),
                    nn.LayerNorm(hparams.mlp_width),
                    nn.LeakyReLU(hparams.mlp_ac),
                    nn.Dropout(hparams.mlp_dropout)
                )
                for _ in range(hparams.mlp_depth - 2)
            ]
        )
        self.output = nn.Linear(hparams.mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        for hidden in self.hiddens:
            x = hidden(x)
        x = self.output(x)
        return x


class Encoder(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams, invertible=False):
        super(Encoder, self).__init__()
        
        if invertible:
            self.mlp = InvertibleMLP(n_inputs, hparams.mlp_width, hparams)
        else:
            self.mlp = MLP(n_inputs, hparams.mlp_width, hparams)

        self.mu = nn.Linear(hparams.mlp_width, n_outputs)
        self.log_var = nn.Linear(hparams.mlp_width, n_outputs)
    
    def reparameterize(self, mu, log_var):
        # 使用重参数化技巧来采样潜在变量z
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.mlp(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        z = self.reparameterize(mu, log_var)
        
        return z, mu, log_var


    
class NotearsDomain(nn.Module):
    def __init__(self, 
                 dims: int, 
                 num_domains: int, 
                 nonlinear: bool = False):
        super(NotearsDomain, self).__init__()
        self.dims = dims
        self.num_domains = num_domains

        init_scale = 0.00001
        self.weight_pos = nn.Parameter(torch.rand(dims + num_domains, dims + num_domains) * init_scale)
        self.weight_neg = nn.Parameter(torch.rand(dims + num_domains, dims + num_domains) * init_scale)

        self.prelu = nn.Sequential(nn.PReLU(), nn.PReLU())
        self.nonlinear = nonlinear

        self.register_buffer("_eye", torch.eye(dims + num_domains))
        
    def _get_adjacency(self, clean_flg: bool = False) -> torch.Tensor:
        """Get adjacency matrix with optional cleaning."""
        if clean_flg:
            W = self.weight_pos[self.num_domains:, self.num_domains:] - self.weight_neg[self.num_domains:, self.num_domains:]
            mask = (1 - self._eye[self.num_domains:, self.num_domains:]).to(W.device)
        else:
            W = self.weight_pos - self.weight_neg
            mask = (1 - self._eye).to(W.device)
       
        return W * mask
    
    def adj_t(self, clean_flg: bool = False) -> torch.Tensor:
        """Get transposed adjacency matrix."""
        W = self._get_adjacency(clean_flg)
        num = W.shape[0] 

        return self._eye[-num:, -num:] - W.t()

    def adj_inv(self, clean_flg: bool = False) -> torch.Tensor:
        """Get inverse of adjacency matrix."""
        return torch.linalg.inv(self.adj_t(clean_flg))

    
    def h_func(self) -> torch.Tensor:
        """Calculate h-score for DAG constraint."""
        W = self._get_adjacency(False)
        E = torch.matrix_exp(W * W)
        return torch.trace(E) - self.dims - self.num_domains

    def w_l1_reg(self) -> torch.Tensor:
        """Calculate L1 regularization term."""
        return torch.mean(self.weight_pos + self.weight_neg)
    
    def l2_reg(self) -> torch.Tensor:
        """Calculate L2 regularization term."""
        weight = self.weight_pos - self.weight_neg
        return torch.sum(weight ** 2)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input features
            y: Optional domain features
            
        Returns:
            Tuple of (processed features, cleaned features)
        """
        W_inv = self.adj_inv(clean_flg=False)
        W_inv_clean = self.adj_inv(clean_flg=True) 

        if y is not None:
            x_aug = torch.cat((y, x), dim=1)
            M = x_aug @ W_inv 
            clean_x = x @ W_inv_clean
        else:
            y = torch.zeros(x.size()[0], self.num_domains).to(x.device)
            x_aug = torch.cat((y, x), dim=1)
            M = x_aug @ W_inv 
            clean_x = x @ W_inv_clean

        if self.nonlinear:
            return self.prelu(M[:, self.num_domains:]), self.prelu(clean_x)
        return M[:, self.num_domains:], clean_x

        
    def mask_feature(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y != None:
            x = torch.cat((y, x), dim=1)
            W = self._get_adjacency()
            x = x @ W.t()
            return self.prelu(x[:, self.num_domains:]) if self.nonlinear else x[:, self.num_domains:]
        W = self._get_adjacency(clean_flg=True)
        x = x @ W.t()
        return self.prelu(x) if self.nonlinear else x


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, domain_dim, hparams, loss_flag='bce'):
        super().__init__()
        self.domain_dim = domain_dim
        self.loss_flag = loss_flag
        # 域判别器
        self.discriminator = MLP(input_dim, domain_dim, hparams)
        
        # 动态权重模块
        self.misclassification_history = torch.zeros(domain_dim)

        if loss_flag == 'bce':
            self.crit = FocalLossV1(alpha=0.25, reduction='none') 
            # nn.BCEWithLogitsLoss(reduction='none')
            # FocalLossV1(alpha=0.5, reduction='none')
        elif loss_flag == 'basebce':
            self.crit = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.crit = nn.MSELoss(reduction='none')
            
    def forward(self, features):
        # 判别器输出
        logits = self.discriminator(features)
        return logits

    def update_weights(self, logits, domain_labels):

        labels = torch.argmax(domain_labels, dim=1)
        cn = logits > logits[labels]

        # 误判率计算
        misclassification_rates = []
        sample_weights = []
        for domain in range(self.domain_dim):
            domain_mask = (labels == domain)
            if domain_mask.float().sum() > 0:
                domain_misclass_rate = cn[domain_mask].float().mean()
            else:
                domain_misclass_rate = torch.tensor(0.0, device=cn.device)
            # domain_misclass_rate = cn[domain_mask].float().mean() if domain_mask.float().sum() > 0 else 0
            misclassification_rates.append(domain_misclass_rate)

        current_misclass_rates = torch.tensor(misclassification_rates)

        self.misclassification_history = (
            0.9 * self.misclassification_history + 
            0.1 * current_misclass_rates
        )
 
        difficulty_weights = F.softmax(
            (1 + self.misclassification_history), 
            dim=0
        )

        return difficulty_weights

    def cal_disc_loss(self, features, domain_labels):
        # 判别器输出
        logits = self.discriminator(features)
        
        # 计算损失
        loss = self.crit(logits, domain_labels)

        # 
        if self.loss_flag == 'bce':
            self.domain_weights = self.update_weights(logits, domain_labels).to(features.device)
            weighted_loss = loss * self.domain_weights 
        elif self.loss_flag == 'basebce':
            weighted_loss = loss
        else:
            weighted_loss = loss
        
        return weighted_loss.sum(axis=-1).mean()
    
