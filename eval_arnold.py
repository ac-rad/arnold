import os
import numpy as np
import yaml
import torch
import torch.optim as optim
import torch.optim as optim
from peract_dataset import PerActDataset
from srt.utils.common import init_ddp
from ddp_sampler_wrapper import DistributedSamplerWrapper
from model import SRT
import sys
sys.path.append('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset')
from arnold_dataset.dataset import ArnoldDataset, prepare_batch


def main():
    rank, world_size = 0, 1
    BC = 'RVT'
    batch_size = 1
    equal_task = False
    cfg = {}
    load_checkpoint = '/home/chemrobot/Documents/RichardHanxu2023/new_kp_runs/runs/peract_cube/ft_c_18_gt_RVT_10/model.pt'
    
    device = torch.device(f"cuda:{rank}")
    
    tasks = [
    #         'stack_blocks',
            'open_drawer',
    #         'meat_off_grill',
    #         'place_shape_in_shape_sorter',
    #         'light_bulb_in',
    #         'turn_tap',
    #         'sweep_to_dustpan_of_size',
    #         'put_money_in_safe',
    #         'stack_cups',
    #         'put_item_in_drawer',
    #         'place_wine_at_rack_location',
    #         'close_jar',
    #         'slide_block_to_color_target',
    #         'reach_and_drag',
    #         'place_cups',
    #         'put_groceries_in_cupboard',
    #         'push_buttons',
    #         'insert_onto_square_peg'
    ]
    task = 'open_drawer'

    print("dataset")
    
    # train_dataset = PerActDataset('peract_dataset', tasks, 'train', BC=BC is not None, no_lang=no_language)
    # val_dataset = PerActDataset('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/peract_dataset', tasks, 'test', BC=BC is not None, no_lang=False)

    dataset = ArnoldDataset('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_re_rendered/open_drawer/train', task, cfg, canonical=True)
    train_sampler = val_sampler = None
    shuffle = True
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True,num_workers=min(batch_size,os.cpu_count()//world_size), sampler=train_sampler, shuffle=shuffle, persistent_workers=True)
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,num_workers=min(batch_size,os.cpu_count()//world_size), sampler=train_sampler, shuffle=shuffle, persistent_workers=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, pin_memory=True,num_workers=min(batch_size,os.cpu_count()//world_size), sampler=val_sampler, shuffle=shuffle, persistent_workers=True)

    print("model")
    LOG_FREQ = 50
    srt = SRT(BC=BC)
    if len(load_checkpoint) > 0 :
        print("load checkpoint")
        stat_dict = srt.load_checkpoint(load_checkpoint)
    
    print('to gpu')
    srt.to(device)
    print('to ddp')
    if world_size > 1:
        srt = torch.nn.parallel.DistributedDataParallel(srt, device_ids=[device])
         
    results = [[], [], []]
    metric_val = 0.0

    # print("PERACT")
    # for batch_ndx, sample in enumerate(val_loader):
    #     batch = {k: v.to(device, non_blocking = True, dtype=torch.float) for k, v in sample.items() if type(v) == torch.Tensor}
    #     print(batch.keys())
    #     for key in batch.keys():
    #         print(batch[key].shape)
    #     loss_terms = srt.eval_step(batch)
    #     metric_val -= loss_terms['trans'].sum() + loss_terms['rot_grip'].sum() + loss_terms['collision'].sum()
    #     metric_name = 'sum_trans_rot'
    #     return
    
    
    print ("ARNOLD")
    
    for batch_ndx, sample in enumerate(loader):
        batch = {k: v.to(device, non_blocking = True, dtype=torch.float) for k, v in sample.items() if type(v) == torch.Tensor}
        batch["language"] = sample["language"]
        batch['ignore_collisions'] = torch.zeros(batch_size, 1).to(device).long()
        batch = prepare_batch(dataset, batch)
        loss_terms = srt.eval_step(batch)
        metric_val -= loss_terms['trans'].sum() + loss_terms['rot_grip'].sum() + loss_terms['collision'].sum()
        metric_name = 'sum_trans_rot'
        print(loss_terms)
        
    metric_val /= len(dataset)/batch_size
    print(f'Validation metric ({metric_name}): {metric_val:.4f}')
    

    # sample = dataset.sample(cfg.batch_size)
    # batch = {k: v.to(device, non_blocking = True, dtype=torch.float) for k, v in sample.items() if type(v) == torch.Tensor}
    # batch['ignore_collisions'] = torch.zeros(cfg.batch_size, 1).to(device)
    # print(batch.keys())
    # for key in batch.keys():
    #     print(batch[key].shape)
    # loss_terms = srt.eval_step(batch)
    # print(loss_terms)


if __name__ == '__main__':
    main()
