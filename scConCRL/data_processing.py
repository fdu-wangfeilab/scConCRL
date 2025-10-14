import numpy as np
import torch
import scanpy as sc 

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.model_selection import train_test_split
from scConCRL.utils import *
import os


class SCData:
    def __init__(self, opt):
        self.opt = opt

    def load_data(self, file_name=None):
        if file_name is None:
            file_name = 'demo.h5ad'
        self.data_path = os.path.join(self.opt.data_dir, self.opt.data_name, 
                                    self.opt.version)
        self.adata_path = os.path.join(self.data_path, file_name)
        self.adata = sc.read_h5ad(self.adata_path)

        self.process_data(is_raw_data=self.opt.is_raw_data, 
                            batch_name=self.opt.batch_name,
                            batch_scale_flag=self.opt.batch_scale)
        self.process_cov(cov_type=self.opt.cov_type)

    def process_data(self, is_raw_data, 
                     batch_name, batch_scale_flag=True, 
                     n_top_features=None):
        
        if is_raw_data:
            self.adata.layers['counts'] = self.adata.X
            sc.pp.filter_cells(self.adata, min_genes=0) # 200
            sc.pp.filter_genes(self.adata, min_cells=0) # 3
            sc.pp.normalize_total(self.adata, target_sum=10000)
            sc.pp.log1p(self.adata) 

        # 此处 n_top_features 是 2000
        if type(n_top_features) == int and n_top_features > 0:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_features, 
                                        batch_key=batch_name, inplace=False, subset=True)

        if batch_scale_flag:
            self.adata.X = batch_scale(self.adata, batch_name)
        else:
            print("scale start!")
            scaler = MaxAbsScaler(copy=False).fit(self.adata.X)
            self.adata.X = scaler.transform(self.adata.X)


    def process_cov(self, cov_type='batch'):
        if cov_type == 'batch':
            n_domain = len(set(self.adata.obs[self.opt.batch_name]))
            domain_mapping = {str(x):np.eye(n_domain)[i] for i, x in enumerate(set(self.adata.obs[self.opt.batch_name]))}
            domain_list = self.adata.obs[self.opt.batch_name].astype(str).apply(lambda x: domain_mapping.get(x, 0)).tolist()
            self.domain_mapping = domain_mapping
            self.cov = np.stack(domain_list)
        elif cov_type == 'condition':
            n_domain = len(set(self.adata.obs[self.opt.condition_name]))
            domain_mapping = {str(x):np.eye(n_domain)[i] for i, x in enumerate(set(self.adata.obs[self.opt.condition_name]))}
            domain_list = self.adata.obs[self.opt.condition_name].astype(str).apply(lambda x: domain_mapping.get(x, 0)).tolist()
            self.domain_mapping = domain_mapping
            self.cov = np.stack(domain_list)
        elif cov_type == 'position':
            scaler = MaxAbsScaler(copy=False).fit(self.adata.obsm['spatial'])
            self.cov = scaler.transform(self.adata.obsm['spatial'])
           
    def split_data(self, split_mode='', test_size=0.1, 
                    seed=1, prediction_type=None):
        
        available_splits = ['simulation', 'no_test', 'no_split', 'ood', 'zeroshot']
        if split_mode not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split_mode = split_mode
        self.seed = seed

        if split_mode in ['no_test']:
            train_idx, val_idx = train_test_split(list(range(self.adata.shape[0])), 
                                                 test_size=0.2, random_state=seed)
            test_idx =  range(self.adata.shape[0])
            self.train_idx = list(train_idx)
            self.val_idx = list(val_idx)
            self.test_idx = list(test_idx)

        elif split_mode in ['no_split']:
            self.train_idx = list(range(self.adata.shape[0]))
            self.val_idx = list(range(self.adata.shape[0]))
            self.test_idx = list(range(self.adata.shape[0]))
        
        elif split_mode in ['ood']:
            if prediction_type is None:
                raise ValueError('prediction_type is required for ood split')
            # 获取训练和测试数据的布尔掩码
            train_mask = self.adata.obs[self.opt.type_name] != prediction_type
            test_mask = self.adata.obs[self.opt.type_name] == prediction_type
            
            # 使用布尔掩码获取索引位置
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            self.train_idx, self.val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)
            self.test_idx = test_idx
        
        elif split_mode in ['zeroshot']:
            if prediction_type is None:
                raise ValueError('prediction_type is required for ood split')
            
            # 获取训练和测试数据的布尔掩码
            train_mask = self.adata.obs[self.opt.batch_name] != prediction_type
            test_mask = self.adata.obs[self.opt.batch_name] == prediction_type

            train_idx = np.where(train_mask)[0].tolist()
            test_idx = np.where(test_mask)[0].tolist()

            self.train_idx, self.val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
            self.test_idx = test_idx    

            # change the cov
            self.domain_mapping[str(prediction_type)] = np.array([0.] * len(self.domain_mapping))
            domain_list = self.adata.obs[self.opt.batch_name].astype(str).apply(lambda x: self.domain_mapping.get(x, 0)).tolist()
            non_zero_columns = np.any(np.array(domain_list) != 0, axis=0)
            self.cov = np.array(domain_list)[:, non_zero_columns]

            
    def get_dataloader(self, batch_size=512, is_shuffle=True):
        
        train_dataset = SingleCellDataset(self.adata[self.train_idx], self.cov[self.train_idx])
        val_dataset = SingleCellDataset(self.adata[self.val_idx], self.cov[self.val_idx])
        test_dataset = SingleCellDataset(self.adata[self.test_idx], self.cov[self.test_idx])
        
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=is_shuffle
        )

        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=is_shuffle
        )
        
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=is_shuffle
        )


def batch_scale(adata, batch_name='batch'):
    """
    Batch-specific scale data
    
    Parameters
    ----------
    adata: AnnData
        Single cell data.
    batch_name: str
        Batch name in adata. DEFAULT: 'batch'
    
    Returns
    -------
    np.array
        Scaled data.
    """
    for b in adata.obs[batch_name].unique():
        idx = np.where(adata.obs[batch_name]==b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        adata.X[idx] = scaler.transform(adata.X[idx])
    return adata.X


class SingleCellDataset(Dataset):
    """
    Dataloader of single-cell data
    """
    def __init__(self, exps, cov):
        self.exps = exps.X.toarray() if sparse.issparse(exps.X) else exps.X
        self.cov = cov
        self.shape = exps.shape

    def __len__(self):
        return self.exps.shape[0]
    
    def __getitem__(self, idx):
        x = self.exps[idx]    
        domain_id = self.cov[idx]
        return x, domain_id, idx


# a function convert anndata to dataloader
def convert_adata_to_dataloader(adata, domain_list, val_split=0.1, batch_size=128, is_shuffle=True, seed=0):
    # simple converter
    
    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X
    else:
        adata.X = adata.X.toarray()
    batch = domain_list.values
     
    dataset = SingleCellDataset(adata, batch)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size != 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_dataset = dataset
        val_dataset = dataset
        
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=is_shuffle
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=is_shuffle
    )
    
    return train_dataloader, val_dataloader
 


def read_data_for_OOD(opt):
    data_format = opt.data_name.split('.')[-1]
    
    if data_format == 'h5ad':
        adata = sc.read_h5ad(opt.data_dir + opt.data_name)
        
    
    # 如果不是稀疏矩阵的格式则转换为稀疏矩阵
    # if not isinstance(adata.X, csr_matrix):
    #     tmp = scipy.sparse.csr_matrix(adata.X)
    #     adata.X = None
    #     adata.X = tmp
    
    if opt.is_raw_data:
        sc.pp.filter_cells(adata, min_genes=0)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata) # 减少数据范围
        
        # 保存一份数据在 adata.raw 中
        adata.raw = adata
        

    dropout_mask = (adata.X != 0).astype(float)
    
    # 此处 n_top_features 是 2000
    if type(opt.n_top_features) == int and opt.n_top_features > 0:
        # 此处是取 2000 个HVG 
        sc.pp.highly_variable_genes(adata, n_top_genes=opt.n_top_features, batch_key=opt.batch_name, inplace=False, subset=True)
    

    if opt.batch_scale:
        print("batch scale start!")
        adata.X = batch_scale(adata, opt.batch_name)
    else:
        print("scale start!")
        scaler = MaxAbsScaler(copy=False).fit(adata.X)
        adata.X = scaler.transform(adata.X)
    
    return adata, dropout_mask

    
def get_cell_type(opt):
    adata = sc.read_h5ad(os.path.join(opt.data_dir, opt.data_name, opt.version, 'demo.h5ad'))
    type_list = list(set(adata.obs[opt.type_name]))
    return type_list

    
def get_batch_type(opt):
    adata = sc.read_h5ad(os.path.join(opt.data_dir, opt.data_name, opt.version, 'demo.h5ad'))
    batch_list = list(set(adata.obs[opt.batch_name]))
    return batch_list

def prepare_data_for_OOD(opt, prediction_type):
    
    # read adata
    adata, dropout_mask = read_data_for_OOD(opt)
    
    n_domain = len(set(adata.obs[opt.condition_name]))
    
    train_adata = adata[adata.obs[opt.type_name] != prediction_type]
    train_domain_list = train_adata.obs[opt.condition_name].apply(lambda x: int(x))

    test_adata = adata[adata.obs[opt.type_name] == prediction_type]
    test_domain_list = test_adata.obs[opt.condition_name].apply(lambda x: int(x))

    return train_adata, train_domain_list, test_adata, test_domain_list, n_domain
