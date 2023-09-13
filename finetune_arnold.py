import os
import numpy as np
import yaml
import torch
import torch.optim as optim
import torch.optim as optim
from srt.utils.common import init_ddp
from ddp_sampler_wrapper import DistributedSamplerWrapper
from model import SRT
import sys
sys.path.append('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset')
from arnold_dataset.dataset import ArnoldDataset, prepare_batch
from lamb import Lamb
import wandb

class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """
    def __init__(self, peak_lr=4e-4, peak_it=10000, decay_rate=0.5, decay_it=100000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:  # Warmup period
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))
    
def check_and_make(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def main():
    rank, world_size = 0, 1
    BC = 'RVT'
    batch_size = 1
    equal_task = False
    cfg = {}
    load_checkpoint = '/home/chemrobot/Documents/RichardHanxu2023/peract_github/peract/ckpts/multi/SRTACT_BC/seed0/weights/BeT1_18/model_best_58596.pt'
    # no_language = False
    
    device = torch.device(f"cuda:{rank}")

    task = 'open_drawer'
    lr = 1e-6
    srt_lr_ratio = None

    print("ARNOLD Dataset")
    
    train_dataset = ArnoldDataset('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_re_rendered/open_drawer/train', task, cfg)
    val_dataset = ArnoldDataset('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_re_rendered/open_drawer/val', task, cfg)
    print(len(train_dataset))

    train_sampler = val_sampler = None
    shuffle = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,num_workers=min(batch_size,os.cpu_count()//world_size), sampler=train_sampler, shuffle=shuffle, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,num_workers=min(batch_size,os.cpu_count()//world_size), sampler=val_sampler, shuffle=shuffle, persistent_workers=True)

    print("model")
    LOG_FREQ = 50
    srt = SRT(BC=BC, freeze_srt=False)
    if len(load_checkpoint) > 0 :
        print("load checkpoint")
        stat_dict = srt.load_checkpoint(load_checkpoint)
    
    print('to gpu')
    srt.to(device)
    print('to ddp')
    if world_size > 1:
        srt = torch.nn.parallel.DistributedDataParallel(srt, device_ids=[device])
    
    print("lr schedule")
    lr_scheduler = LrScheduler(peak_lr=1e-5, peak_it=2500, decay_it=4000000, decay_rate=0.16)
    optimizer = Lamb(srt.parameters(), lr=lr, weight_decay=0.000001, betas=(0.9, 0.999), adam=False)

    resume = True
    itr = 0
    if resume:
        optimizer.load_state_dict(stat_dict['optimizer'])
        itr = int(load_checkpoint.split('.')[0].split('_')[-1])
        print(itr)
    
    metric_val_best = -np.inf
    backup_every = 20
    validate_every = 10
    print_every = 10
    avg_loss = {"trans": 0,
                "rot_grip": 0,
                "collision": 0,
                "render": 0}

    epochs = 10
    save_folder = 'runs/fine_tune_rvt_open_drawer/'

    for eps in range(epochs):
        if world_size>1:
            train_sampler.set_epoch(eps)
        for batch_ndx, sample in enumerate(train_loader):
            checkpoint_scalars = {'it': itr,'loss_val_best': metric_val_best}
            new_lr = lr_scheduler.get_cur_lr(itr)
            for param_group in optimizer.param_groups:
                if srt_lr_ratio:
                    if 'pose_decoder' in param_group['name']:
                        param_group['lr'] = new_lr
                    else:
                        param_group['lr'] = new_lr/srt_lr_ratio
                else:
                    param_group['lr'] = new_lr
            if world_size > 1:
                srt.module.train()
            else:
                srt.train()
            optimizer.zero_grad()
            batch = {k: v.to(device, non_blocking = True, dtype=torch.float) for k, v in sample.items() if type(v) == torch.Tensor}
            batch["language"] = sample["language"]
            batch['ignore_collisions'] = torch.zeros(batch_size, 1).to(device)
            batch = prepare_batch(train_dataset, batch)

            loss, loss_term = srt(batch)
            loss = loss.mean(0)
            loss_term = {k: v.sum().item() for k, v in loss_term.items()}
            loss.backward()
            optimizer.step()
            if rank == 0:
                if itr % backup_every == 0:
                    if world_size > 1:
                        srt.module.save_checkpoint(f"{save_folder}/model.pt", checkpoint_scalars, optimizer)
                    else:
                        srt.save_checkpoint(f"{save_folder}/model.pt", checkpoint_scalars, optimizer)
                for k, v in loss_term.items():
                    avg_loss[k] += v
                if itr % print_every == 0:
                    for k in avg_loss.keys():
                        avg_loss[k] /= print_every
                    print(avg_loss)
                    # if rank == 0:
                    #     wandb.log(avg_loss, step=itr*batch_size*world_size)
                    for k in avg_loss.keys():
                        avg_loss[k] = 0
            itr += 1
        
        print('Evaluating...')
        metric_val = 0
        for batch_ndx, sample in enumerate(val_loader):
            if world_size > 1:
                srt.module.eval()
            else:
                srt.eval()
            batch = {k: v.to(device, non_blocking = True, dtype=torch.float) for k, v in sample.items() if type(v) == torch.Tensor}
            batch["language"] = sample["language"]
            batch['ignore_collisions'] = torch.zeros(batch_size, 1).to(device)
            batch = prepare_batch(val_dataset, batch)

            if world_size > 1:
                eval_dict = srt.module.eval_step(batch)
            else:
                eval_dict = srt.eval_step(batch)
            if BC is not None:
                metric_val -= eval_dict['trans'].sum() + eval_dict['rot_grip'].sum() + eval_dict['collision'].sum()
                metric_name = 'sum_trans_rot'
            else:
                metric_val += eval_dict['psnr'].sum()
                metric_name = 'psnr'
        
        metric_val /= len(val_dataset)/batch_size
        print(f'Validation metric ({metric_name}): {metric_val:.4f}/{metric_val_best:.4f}')
        # if rank == 0:
        #     wandb.log({f"{metric_name}": metric_val, 'current_best':metric_val_best},step=itr*batch_size*world_size)
        if (metric_val - metric_val_best) > 0:
            metric_val_best = metric_val
            if rank == 0:
                checkpoint_scalars['loss_val_best'] = metric_val_best
                print(f'New best model (loss {metric_val_best:.6f})')
                if world_size > 1:
                    srt.module.save_checkpoint(f"{save_folder}/model_best.pt", checkpoint_scalars, optimizer)
                else:
                    srt.save_checkpoint(f"{save_folder}/model_best.pt", checkpoint_scalars, optimizer)


if __name__ == '__main__':
    main()
