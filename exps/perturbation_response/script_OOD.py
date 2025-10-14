
import sys
sys.path.append('/home/dongjiayi/workbench/denoise/scConCRL')

import scanpy as sc
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
import tqdm
import pandas as pd
import argparse

from omegaconf import OmegaConf

from scConCRL.utils import *
from scConCRL.model import *
from scConCRL.data_processing import *
from scConCRL.eval import *

warnings.filterwarnings(action='ignore')

def train_model(opt, prediction_type):
    set_seed(opt.reproduc.seed)

    scdata = SCData(opt=opt.data)
    scdata.load_data()
    scdata.split_data(split_mode='ood', prediction_type=prediction_type)
    scdata.get_dataloader()

    # setting training
    input_dim = scdata.adata.X.shape[1]
    cov_dim = scdata.cov.shape[1]

    mdl = model(
        input_dim = input_dim,
        domain_dim = cov_dim,
        hparams = opt.model,
    )

    mdl.train(train_dataloader=scdata.train_dataloader, 
                val_dataloader=scdata.val_dataloader, hparams=opt.model,
                output_path=os.path.join(opt.output_path, opt.data.data_name, prediction_type)
                )

    train_adata = mdl.data_transform(
        dataloader=scdata.train_dataloader, 
        data=scdata.adata[scdata.train_idx].copy(),
        domain_id=scdata.domain_mapping['1'],
        clip_data=False, 
    )

    val_adata = mdl.data_transform(
        dataloader=scdata.val_dataloader, 
        data=scdata.adata[scdata.val_idx].copy(),
        domain_id=scdata.domain_mapping['1'],
        clip_data=False, 
    )

    train_pred_data = concat_ctrl_case_pred_data(sc.concat([train_adata, val_adata]))


    test_adata = mdl.data_transform(
        dataloader=scdata.test_dataloader, 
        data=scdata.adata[scdata.test_idx].copy(),
        domain_id=scdata.domain_mapping['1'],
        clip_data=False, 
    )

    test_pred_data = concat_ctrl_case_pred_data(test_adata)
    return test_pred_data


def main(opt):
    set_seed(opt.reproduc.seed)

    type_list = get_cell_type(opt.data)
    concat_adata_list = []
    for prediction_type in type_list:
        concat_adata = train_model(opt, prediction_type)
        concat_adata_list.append(concat_adata)

    concat_adata = sc.concat(concat_adata_list)  
    visualize_data(concat_adata, colors=[opt.data.condition_name, opt.data.type_name], 
                    save_path=os.path.join(opt.output_path, opt.data.data_name, 
                    'figures'), save_name=f'_ood_tcc_{opt.data.data_name}')     
    
    conditions_map = {
        '0': 'ctrl',
        '1': 'case',
        '2': 'pred'
    }

    ood_metrics(concat_adata, type_list, conditions_map, os.path.join(opt.output_path, opt.data.data_name), type_name=opt.data.type_name)

    concat_adata.write(os.path.join(opt.output_path, opt.data.data_name, f'ood_{opt.reproduc.seed}.h5ad'))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training')
    # 添加位置参数
    parser.add_argument('--opt_file', default="../../config/demo_OOD.yaml", type=str, help='opt file')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    args = parser.parse_args()

    opt = OmegaConf.load(args.opt_file) # 加载配置文件
    opt.reproduc.seed = args.seed
    main(opt)

