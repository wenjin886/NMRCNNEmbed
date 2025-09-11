import torch
from torch.utils.data import Dataset, DataLoader

import json
from argparse import ArgumentParser
from datetime import datetime
import random
from tqdm import tqdm

import os
os.environ.pop("SLURM_NTASKS", None)
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from dataset_dataloader import prep_ir_geo_dataset, qm9_dataset_collate_fn, qm9_dataset_info
from nmr_cnn_encoder import NMR_CNN_Encoder_PeakPrediction

def get_formatted_exp_name(exp_name, resume=False):
    formatted_time = datetime.now().strftime("%H-%M-%d-%m-%Y")
    if resume and ("resume" not in exp_name):
        formatted_exp_name = f"resume_{exp_name}_{formatted_time}"
    else:
        formatted_exp_name = f"{exp_name}_{formatted_time}"
    return formatted_exp_name

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(args.precision)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = torch.float32

    dataset_dic = prep_ir_geo_dataset(args.data_dir, dataset='qm9', test_run=args.code_test)
    

    if args.code_test:
        dataloader = DataLoader(dataset_dic['val'], batch_size=args.batch_size, collate_fn=qm9_dataset_collate_fn,
                                shuffle=False, num_workers=17)
    else:
        train_dataloader = DataLoader(dataset_dic['train'], batch_size=args.batch_size, collate_fn=qm9_dataset_collate_fn,
                                shuffle=True, num_workers=17)
        val_dataloader = DataLoader(dataset_dic['val'], batch_size=args.batch_size, collate_fn=qm9_dataset_collate_fn,
                                shuffle=False, num_workers=17)
        test_dataloader = DataLoader(dataset_dic['test'], batch_size=args.batch_size, collate_fn=qm9_dataset_collate_fn,
                                shuffle=False, num_workers=17)
        
        exp_name = get_formatted_exp_name(args.exp_name, resume=args.resume)
        save_dirpath = osp.join(args.exp_save_path, exp_name)
        print(f"Will save training results in {save_dirpath}")
        if not os.path.exists(save_dirpath):
            print(f"Making dir: {save_dirpath}")
            os.makedirs(save_dirpath)
    
    model = NMR_CNN_Encoder_PeakPrediction(enc_output_dim=512, 
                                             max_num_peaks=qm9_dataset_info['max_num_peaks_H'],
                                             one_hot_dim=qm9_dataset_info['max_num_atoms_peak_H']+1, 
                                             ppm_min=-5, ppm_max=20,
                                             lr=args.lr, warm_up_step=args.warm_up_step,
                                             device=device, dtype=dtype)
    model.to(device)  

    

    if args.code_test:
        wandb_logger = None
        fast_dev_run = 2
        limit_val_batches, limit_test_batches = 0, 0
        print(model)
    else:
        wandb_logger = WandbLogger(
                    project=args.wandb_project,
                    name=exp_name,
                    save_dir=save_dirpath
                )
        wandb_logger.experiment.config.update({"model_arch": str(model)})
        fast_dev_run = False
        limit_val_batches, limit_test_batches = 1.0, 1.0


    checkpoint_callback = ModelCheckpoint(dirpath=save_dirpath, 
                                          save_top_k=args.save_top_k, 
                                          every_n_epochs=args.save_every_n_epochs,
                                          monitor=args.monitor,
                                          mode=args.monitor_mode,
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.early_stop != -1:
        early_stop_callback = EarlyStopping(
                monitor=args.monitor,      # 监控指标，例如验证集损失
                patience=args.early_stop,             # 如果n个epoch内指标未改善则停止训练
                mode=args.monitor_mode,              # 监控指标越小越好（如损失函数）
                verbose=True             # 是否打印信息
            )
        callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
    else:
        callbacks = [checkpoint_callback, lr_monitor]
    
    if device == torch.device("cuda"):
        device_num = torch.cuda.device_count()
        accelerator = "gpu"
    else:
        device_num = "auto"
        accelerator = "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device_num,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        fast_dev_run=fast_dev_run,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        callbacks=callbacks
    )   
    
    
    # 在code_test模式下直接测试，不使用checkpoint
    if args.code_test:
        trainer.fit(model, dataloader)
    else:
        
        with open(osp.join(save_dirpath, 'config.json'), 'w') as f: # 保存为 JSON 文件
            args_dict = vars(args)
            json.dump(args_dict, f, indent=4)

        trainer.fit(model, train_dataloader, val_dataloader,
                ckpt_path=args.checkpoint_path if args.resume else None
            )
        trainer.test(ckpt_path="best", dataloaders=test_dataloader)







if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--exp_save_path', type=str, default='../exp')
    parser.add_argument('--data_dir', type=str, default='/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_IR/DATASET/qm9/qm9-nmr/qm9_nmr_split')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str)


    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--warm_up_step', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--early_stop', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    
    parser.add_argument('--wandb_project', type=str, default='NMRembedding')
    parser.add_argument('--precision', type=str, default='medium', choices=['medium', 'high'])
    parser.add_argument('--monitor', type=str, default='val_loss')
    parser.add_argument('--monitor_mode', type=str, default='min', choices=['min', 'max'])
    
    args = parser.parse_args()
    main(args)