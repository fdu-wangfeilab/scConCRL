
import sys
import os
sys.path.append('/home/dongjiayi/workbench/denoise/scConCRL')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import scanpy as sc
import numpy as np
import torch
import os
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

def main(opt):
    set_seed(opt.reproduc.seed)

    scdata = SCData(opt=opt.data)
    scdata.load_data()
    scdata.split_data(split_mode='no_split')
    scdata.get_dataloader()

    # setting training
    input_dim = scdata.adata.X.shape[1]
    cov_dim = scdata.cov.shape[1]

    mdl = model(
        input_dim = input_dim,
        domain_dim = cov_dim,
        hparams = opt.model,
    )

    mdl.train(
                train_dataloader=scdata.train_dataloader, 
                val_dataloader=scdata.val_dataloader,
                hparams=opt.model,
                output_path=os.path.join(opt.output_path, opt.data.data_name)
                )

    data = mdl.evaluate(
        dataloader=scdata.test_dataloader,
        data=scdata.adata,
    )

    grn_metrics = grn_inference_metrics(mdl.grn_inference(), 
                                        scdata.adata.var_names.to_series().apply(lambda x: x.upper()),
                                        os.path.join(opt.data.data_dir, opt.data.data_name, opt.data.version),
                                        sub_task=opt.sub_task,)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training')
    # 添加位置参数
    parser.add_argument('--opt_file', default="../../config/demo_grn.yaml", type=str, help='opt file')
    parser.add_argument('--version', default="v2", type=str, help='version')
    parser.add_argument('--alpha', default=1, type=int, help='alpha') 
    parser.add_argument('--dag_anneal_steps', default=100, type=int, help='lambda_kl2')
    parser.add_argument('--adv_step', default=10, type=int, help='lambda_kl2')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--epochs', default=600, type=int, help='epochs')
    parser.add_argument('--sub_task', default='lofgof', type=str, help='sub_task')
    args = parser.parse_args()

    opt = OmegaConf.load(args.opt_file) # 加载配置文件
    opt.model.alpha = args.alpha
    opt.reproduc.seed = args.seed
    opt.data.version = args.version
    opt.model.epochs = args.epochs
    opt.sub_task = args.sub_task
    if args.sub_task in ['Spec', 'lofgof']:
        opt.model.lambda_kl1 = 0.1
        opt.model.lambda_kl2 = 0.1
        opt.model.lambda_kl3 = 0.0001
        opt.model.lambda_l2 = 0.03
        opt.model.lambda_l1 = 0.03
    elif args.sub_task in ['NSpec', 'STR']:
        opt.model.lambda_kl1 = 0.0001
        opt.model.lambda_kl2 = 0.0001
        opt.model.lambda_kl3 = 0.0001
        opt.model.lambda_l2 = 0.01
        opt.model.lambda_l1 = 0.01
    main(opt)

