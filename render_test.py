import hydra
from omegaconf import OmegaConf

import os
import torch
import torch.nn as nn
import sys
import numpy as np
from environment.runner_utils import get_simulation
import matplotlib.pyplot as plt
import time

def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []
    fnames = []

    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
        fnames.append(npz_path)
    return data, fnames

def main(cfg):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device is {device}')
  task_list = ['open_drawer']

  simulation_app, simulation_context, new_simulation = get_simulation(headless = False)
  from arnold_dataset.tasks import load_task
  i = 0
  is_done = False

  while simulation_app.is_running():
    if not is_done:
      is_done = True
      for task in task_list:
        # data, fnames = load_data(data_path=os.path.join('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset/configs', task, 'val'))
          anno = np.load("/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset/data/open_drawer/test/Steven-open_drawer-0-0-0.0-0.5-2-Mon_Jan_30_06:06:37_2023.npz", allow_pickle=True)
      #   anno = data.pop(0)
      #   fname = fnames.pop(0)
          gt_frames = anno['gt']
          robot_base = gt_frames[0]['robot_base']
          gt_actions = [
                          gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world'],
                          gt_frames[3]['position_rotation_world'] if 'water' not in task \
                          else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                      ]

          env, object_parameters, robot_parameters, scene_parameters = load_task('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset/assets', npz=anno, cfg=cfg)
          obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
                        robot_base=robot_base, gt_actions=gt_actions)
          
          obs, suc = env.step(act_pos=None, act_rot=None, render=True, use_gt=True)
          
          i = 0
          print('Keys')
          print(obs.keys())
          for i in range(len(obs['images'])):      
            plt.imsave(f'./image_out/{i}.png', obs['images'][i]['rgb'])
        
          if obs is not None:
            break
    simulation_app.update()
    # break
  
  simulation_app.close()

class DotDict:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, attr):
        if attr in self.d:
            return self.d[attr]
        raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

def hydra_extract():
  import omegaconf
  cfg = omegaconf.OmegaConf.load('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset/configs/default.yaml')
  return dict(cfg)

if __name__ == '__main__':
  s = hydra_extract()
  s = DotDict(s)
  main(s)