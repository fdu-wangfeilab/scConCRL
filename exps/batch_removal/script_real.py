
import sys
sys.path.append('/home/dongjiayi/workbench/denoise/scConCRL')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
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
    scdata.split_data(split_mode='no_test')
    scdata.get_dataloader()

    # setting training
    input_dim = scdata.adata.X.shape[1]
    cov_dim = scdata.cov.shape[1]
    output_path = os.path.join(opt.output_path, opt.data.data_name)
    
    mdl = model(
        input_dim = input_dim,
        domain_dim = cov_dim,
        hparams = opt.model,
        # invertible=True, # invertible
    )

    if opt.model.load_model_flag:
        mdl.load(output_path)       
    else:
        mdl.train(train_dataloader=scdata.train_dataloader, 
                val_dataloader=scdata.val_dataloader,
                    hparams=opt.model,
                    output_path=output_path,
                    )

    data = mdl.evaluate(
        dataloader=scdata.test_dataloader,
        data=scdata.adata,
    )


    visualize_diff_data(scdata.adata, colors=[opt.data.batch_name, opt.data.state_name],
                    save_path=os.path.join(opt.output_path, opt.data.data_name))
    
    test_metrics = batch_removal_metrics(scdata.adata,
                                        opt.data.batch_name, 
                                        opt.data.state_name,
                                        save_path=output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training')
    # 添加位置参数
    parser.add_argument('--opt_file', default="../../config/demo_denoise.yaml", type=str, help='opt file')
    parser.add_argument('--dataset', default='hPancreas', type=str, help='dataset')
    parser.add_argument('--lambda_align', default=30, type=float, help='lambda_align')
    parser.add_argument('--lambda_kl1', default=0.0001, type=float, help='lambda_kl1')
    parser.add_argument('--lambda_kl2', default=0.01, type=float, help='lambda_kl2')
    parser.add_argument('--lambda_kl3', default=0.01, type=float, help='lambda_kl3')
    parser.add_argument('--alpha', default=1, type=int, help='alpha')
    parser.add_argument('--dag_anneal_steps', default=100, type=int, help='lambda_kl2')
    parser.add_argument('--adv_step', default=10, type=int, help='lambda_kl2')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    args = parser.parse_args()

    opt = OmegaConf.load(args.opt_file) # 加载配置文件
    

    opt.data.data_name = args.dataset
    opt.reproduc.seed = args.seed

    main(opt)

