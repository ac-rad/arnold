import hydra
from omegaconf import OmegaConf

import os
import torch
import torch.nn as nn
import sys
import numpy as np
from pathlib import Path
from environment.runner_utils import get_simulation
import matplotlib.pyplot as plt

SAVE_DIR = '/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_re_rendered'
DATA_DIR = '/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset/data'
SPLIT = 'train'

def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []
    fnames = []

    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
        fnames.append(npz_path)
    return data, fnames

def save_camera_renders(obs, gt_frame, obs_counter):
    for i in range(len(obs['images'])):      
      plt.imsave(f'./image_out/obs_{obs_counter}_cube_{i}.png', obs['images'][i]['rgb'])
          
    for i in range(len(gt_frame['images'])):
      plt.imsave(f'./image_out/obs_{obs_counter}_base_{i}.png', gt_frame['images'][i]['rgb'])


def add_cube_to_observation(obs, gt_frame):
   return np.concatenate((gt_frame['images'], obs['images'][5:]))

def save_observation_np(gt_frame, path):
   np.save(path, gt_frame, allow_pickle=True)

def main(cfg):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device is {device}')
  task_list = ['open_drawer']
  simulation_app, _, _ = get_simulation(headless = True)
  
  from arnold_dataset.tasks import load_task
  
  obs_counter = 0
  episode_counter = 0
  i = 0

  while simulation_app.is_running():
    
    for task in task_list:
      data, fnames = load_data(os.path.join(DATA_DIR, task, SPLIT))
      os.makedirs(os.path.join(SAVE_DIR, task, SPLIT), exist_ok=True)
      print(f"Rendering {len(data)} episodes")

      while len(data) > 0:
        print(f"Episode {i}")
        i += 1
        
        anno = data.pop(0)
        fname = fnames.pop(0)
        gt_frames = anno['gt'].copy()
        robot_base = gt_frames[0]['robot_base']
        gt_actions = [
                        gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world'],
                        gt_frames[3]['position_rotation_world'] if 'water' not in task \
                        else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                    ]
        env, object_parameters, robot_parameters, scene_parameters = load_task('/home/chemrobot/Documents/RichardHanxu2023/SRTACT_Eval/arnold_dataset/assets', npz=anno, cfg=cfg)
        obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
                      robot_base=robot_base, gt_actions=gt_actions)
        
        gt_frames[0]['images'] = add_cube_to_observation(obs, gt_frames[0])
        # save_camera_renders(obs, gt_frames[0], obs_counter)
        
        obs_counter += 1
        for i in range(len(gt_actions)):
          obs, suc = env.step(act_pos=None, act_rot=None, render=True, use_gt=True)
          gt_frames[i + 1]['images'] = add_cube_to_observation(obs, gt_frames[i+1])
          # save_camera_renders(obs, gt_frames[i+1], obs_counter)
          obs_counter += 1
        
        save_observation_np(gt_frames, Path(f'{SAVE_DIR}/{task}/{SPLIT}/{fname.split("/")[-1]}'))
        env.stop()
    
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