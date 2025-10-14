import copy
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from scConCRL.network import MLP, Encoder, NotearsDomain, DomainDiscriminator
from scConCRL.utils import get_optimizer, squared_loss,\
                         kl_loss, tc_loss, EarlyStopping, tensor2numpy, \
                        GradualWarmupScheduler




class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_dim, domain_dim, hparams):
        super(Algorithm, self).__init__()
        self.input_dim = input_dim
        self.domain_dim = domain_dim
        self.num_hidden = hparams.num_hidden

        self.hparams = hparams
        self.lambda_rec = self.hparams.lambda_rec
        self.lambda_kl1 = self.hparams.lambda_kl1
        self.lambda_kl2 = self.hparams.lambda_kl2
        self.lambda_kl3 = self.hparams.lambda_kl3
        self.lambda_tc = self.hparams.lambda_tc
        self.lambda_align = self.hparams.lambda_align
        self.lambda_recon_croase = self.hparams.lambda_recon_croase if hasattr(self.hparams, 'lambda_recon_croase') \
                                    else self.hparams.lambda_rec * 0.1
        
        self.rho_max = self.hparams.rho_max
        self.alpha = self.hparams.alpha
        self.rho = self.hparams.rho
        self.lambda_l2 = self.hparams.lambda_l2
        self.lambda_l1 = self.hparams.lambda_l1
        self._h_val = np.inf
        self.adv_step = self.hparams.adv_step

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        return clone
    
    
    
class model(Algorithm):

    def __init__(self, input_dim, domain_dim, hparams, invertible=False):
        super(model, self).__init__(input_dim,
                                   domain_dim,
                                   hparams,
                                   )
        
        self.feature_encoder = Encoder(input_dim, input_dim, self.hparams, invertible=invertible)
        
        self.domain_encoder = Encoder(input_dim, hparams.num_hidden, self.hparams, invertible=invertible)
        
        self.domain_prior = Encoder(domain_dim, hparams.num_hidden, self.hparams)

        self.dag_mlp = NotearsDomain(input_dim, hparams.num_hidden, nonlinear=self.hparams.nonlinear)
        self.dag_mlp.weight_pos.data[:hparams.num_hidden, hparams.num_hidden:].fill_(0.0) # hparams.num_hidden:
        self.dag_mlp.weight_neg.data[:hparams.num_hidden, hparams.num_hidden:].fill_(0.0)
        self.dag_mlp.weight_pos.data[:hparams.num_hidden, hparams.num_hidden:].requires_grad = False
        self.dag_mlp.weight_neg.data[:hparams.num_hidden, hparams.num_hidden:].requires_grad = False

        self.disc = DomainDiscriminator(input_dim, domain_dim, self.hparams)

        self.decoder = nn.Sequential(nn.PReLU(), nn.PReLU())
        self.network = nn.Sequential(self.feature_encoder, self.domain_encoder, 
                                     self.domain_prior, self.dag_mlp, self.decoder)

        # use GPU for training if cuda is available
        if torch.cuda.is_available():
            self.network = self.network.cuda()
            self.disc = self.disc.cuda()
            self.device = torch.device('cuda')

        if self.hparams.loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()


    def set_train(self):
        self.network.train()
        self.disc.train()

    def set_eval(self):
        self.network.eval()
        self.disc.eval()

    def save_model_dict(self, output_path):
        torch.save(self.state_dict(), output_path + '/model/model_dict.pt')

    def load(self, output_path):
        self.load_state_dict(torch.load(os.path.join(output_path, 'model/model_dict.pt')))
    
    def train(self, 
            train_dataloader, 
            val_dataloader,
            output_path, 
            hparams = None,
            wandb_instance=None):
        step = 0
        
        self.optim = get_optimizer(
            hparams.optimizer,
            [{"params": self.network.parameters()}],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        self.disc_optim = get_optimizer(
            hparams.optimizer,
            [{"params": self.disc.parameters()}],
            lr=self.hparams.lr * 2,
            weight_decay=self.hparams.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.9, patience=10, verbose=True)
        self.scheduler_warmup = GradualWarmupScheduler(self.optim, multiplier=1, total_epoch=20, after_scheduler=self.scheduler)
        self.early_stopping = EarlyStopping(patience=self.hparams.patience, verbose=False)
        self.optim.zero_grad() 
        self.optim.step()

        self.output_path = output_path
        if not os.path.exists(output_path + '/model'):
            os.makedirs(output_path + '/model', exist_ok=True)

        with tqdm(total = self.hparams.epochs, ncols=100) as pbar:
            pbar.set_description('training')
            for epoch in range(self.hparams.epochs):
                train_loss, val_loss, step = self.train_model(train_dataloader, val_dataloader, step)
                pbar.update(1)

                pbar.set_postfix(
                    train='{:.4f}'.format(train_loss['train_loss']), 
                    val='{:.4f}'.format(val_loss['val_loss']))    
                
                self.early_stopping(val_loss['val_recon1'], self, self.output_path)
                self.scheduler.step(val_loss['val_recon1'])

                if wandb_instance is not None:
                    wandb_instance.log(train_loss)
                    wandb_instance.log(val_loss)
                    
                if self.early_stopping.early_stop:
                    print('Integrative training early stop, validation loss does not improve in '+str(self.hparams.patience)+' epoches!')
                    self.load_state_dict(torch.load(os.path.join(self.output_path, 'model/model_dict.pt')))
                    break

    def train_model(self, train_dataloader, val_dataloader, step):
        self.set_train()
        epoch_loss = defaultdict(float)
        val_epoch_loss = defaultdict(float)
        for i, (x, d, idx) in enumerate(train_dataloader):
            exprs = x.float().to(self.device)
            batch = d.float().to(self.device)
            step += 1

            # upadate the discriminator
            for _ in range(self.hparams.adv_step):
                z, mu, log_var = self.feature_encoder(exprs)
                disc_loss = self.disc.cal_disc_loss(mu, batch)
                self.disc_optim.zero_grad()
                disc_loss.backward()
                self.disc_optim.step()

            # upadate the network
            loss = self.calculate_loss(exprs, batch, step)
            self.optim.zero_grad()
            loss['loss'].backward()
            self.optim.step()

            for k, v in loss.items():
                epoch_loss['train_' + k] += loss[k].item()
        epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}        

        self.set_eval()
        for i, (x, d, idx) in enumerate(val_dataloader):
            exprs = x.float().to(self.device)
            batch = d.float().to(self.device)
            loss = self.calculate_loss(exprs, batch, step)
            for k, v in loss.items():
                val_epoch_loss['val_' + k] += loss[k].item()
        val_epoch_loss = {k:v/(i+1) for k, v in val_epoch_loss.items()}

        return epoch_loss, val_epoch_loss, step

    
    def calculate_loss(self, x, batch, step):
        
        # train model
        z, mu, log_var = self.feature_encoder(x)
        d, d_mu, d_log_var = self.domain_encoder(x)
        p, p_mu, p_log_var = self.domain_prior(batch)
        
        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=d_mu)
        rec_x_p_mu, clean_x_p_mu = self.dag_mlp(x=mu, y=p_mu)

        rec_x = self.reparameterize(rec_x_mu, torch.log(0.00001 * torch.ones_like(log_var)))
        rec_x = self.decoder(rec_x)

        meta_cell = x
        rec_u = self.decoder(self.dag_mlp.mask_feature(x=meta_cell))
        
        # reconstruction loss
        loss_rec_1 = squared_loss(rec_x, x) * self.lambda_rec
        loss_rec_2 = squared_loss(rec_u, meta_cell) * self.lambda_recon_croase
        loss_rec = loss_rec_1 + loss_rec_2
        
        loss_kl_1 = kl_loss(mu, log_var) * self.lambda_kl1
        loss_kl_2 = kl_loss(d_mu, d_log_var, p_mu, p_log_var) * self.lambda_kl2
        loss_kl_3 = kl_loss(rec_x_mu, torch.ones_like(log_var), rec_x_p_mu, torch.ones_like(log_var)) * self.lambda_kl3
        loss_kl = loss_kl_1 + loss_kl_2 + loss_kl_3
        
        # dag loss
        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l2_reg = self.lambda_l2 * self.dag_mlp.l2_reg() 
        l1_reg = self.lambda_l1 * self.dag_mlp.w_l1_reg() 

        loss_dag =  penalty + l2_reg + l1_reg 

        # align loss
        loss_g_disc = -self.disc.cal_disc_loss(mu, batch) * self.lambda_align   
        
        loss_rob = tc_loss(mu, d_mu) * self.lambda_tc

        if step >= self.hparams["dag_anneal_steps"]:
            loss = loss_rec + loss_dag + loss_g_disc + loss_rob + loss_kl
        else:
            loss = loss_rec + loss_g_disc + loss_rob + loss_kl
            loss_dag = torch.tensor(0.0)

        
        return {"loss": loss,
                "disc": loss_g_disc,
                "recon1": loss_rec_1,
                "recon2": loss_rec_2,
                "kl1": loss_kl_1,
                "kl2": loss_kl_2,
                "kl3": loss_kl_3,
                "dag": loss_dag,
                "penalty": penalty,
                "l1": l1_reg,
                "l2": l2_reg,
                "tc": loss_rob}
        

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def predict(self, x, batch):
        z, mu, log_var = self.feature_encoder(x)
        d, d_mu, d_log_var = self.domain_encoder(x)
    
        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=d_mu)
        
        rec_x = self.decoder(rec_x_mu)
        clean_x = self.decoder(clean_x_mu)
        
        return rec_x, clean_x, mu, d_mu


    def predict_clean_data(self, x):
        z, mu, log_var = self.feature_encoder(x)
        
        batch = torch.zeros((x.shape[0], self.domain_dim), dtype=torch.float).to(self.device)
        p, p_mu, p_log_var = self.domain_prior(batch)

        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=p_mu)
        clean_x = self.decoder(clean_x_mu)
        return clean_x


    def evaluate(self, dataloader, data, clip_data=True):
        self.set_eval()
        num = data.shape[0]
        dim = data.shape[1]
        denoises = np.zeros((num, dim))
        recons = np.zeros((num, dim))
        latents = np.zeros((num, dim))
        for (x, d, idx) in tqdm(dataloader, total=len(dataloader)):
            x, d = x.float().to(self.device), d.float().to(self.device)
            recon_x, clean_x, z, domain = self.predict(x, d)

            recons[idx] = tensor2numpy(recon_x, clip_data=clip_data)
            denoises[idx] = tensor2numpy(clean_x, clip_data=clip_data)
            latents[idx] = tensor2numpy(z, clip_data=clip_data)
  
        data.obsm['X_denoised'] = denoises
        data.obsm['X_recon'] = recons
        data.obsm['X_latent'] = latents
        return data
    
    def transfer(self, x, domain):
        z, mu, log_var = self.feature_encoder(x)
        domain = torch.tensor(domain).float().to(self.device)
        domain = domain.repeat(x.shape[0], 1)
        d, d_mu, d_log_var = self.domain_prior(domain)
        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=d_mu)

        rec_x = self.decoder(rec_x_mu)
        return rec_x
    
    def data_transform(self, dataloader, data, domain_id, clip_data=True):
        self.set_eval()
        num = data.shape[0]
        dim = data.shape[1]
        trans = np.zeros((num, dim))
        for (x, d, idx) in tqdm(dataloader, total=len(dataloader)):
            x, d = x.float().to(self.device), d.float().to(self.device)
            recon_x = self.transfer(x, domain_id)
            trans[idx] = tensor2numpy(recon_x, clip_data=clip_data)
        data.obsm['X_trans'] = trans
        return data
    
    def data_denoise(self, dataloader, data, clip_data=True):
        self.set_eval()
        num = data.shape[0]
        dim = data.shape[1]
        denoised = np.zeros((num, dim))
        for (x, d, idx) in tqdm(dataloader, total=len(dataloader)):
            x, d = x.float().to(self.device), d.float().to(self.device)
            clean_x = self.predict_clean_data(x)
            denoised[idx] = tensor2numpy(clean_x, clip_data=clip_data)
        data.obsm['X_denoised'] = denoised
        return data

    def grn_inference(self):
        adj = self.dag_mlp._get_adjacency(clean_flg=True).cpu().detach().numpy()
        return adj

    def key_genes_detection(self):
        adj = self.dag_mlp._get_adjacency(clean_flg=False).cpu().detach().numpy()
        adj = adj[self.num_hidden:, :self.num_hidden]
        return adj