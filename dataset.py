import os
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from utils.transforms import *
import clip
from tqdm import tqdm
import einops
from srt.utils.nerf import transform_points_torch, get_extrinsic_torch

CAMERAS = ['front', 'base', 'left', 'wrist_bottom', 'wrist', 'forward', 'top', 'back', 'left', 'right']
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
pickle.DEFAULT_PROTOCOL=pickle.HIGHEST_PROTOCOL
RANDOM_SEED=1125
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class CLIP_encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, preprocess = clip.load("RN50", device=device, jit=False)
    
    @torch.no_grad()
    def encode_text(self, text):
        tokens = clip.tokenize(text)
        tokens = tokens.to(self.device)
        x = self.model.token_embedding(tokens).type(self.model.dtype)   # [B, T, D]

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)   # BTD -> TBD
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)   # TBD -> BTD
        x = self.model.ln_final(x).type(self.model.dtype)

        return x

def point_to_voxel_index(points: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indices = np.minimum(np.floor((points - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one)
    return voxel_indices

def normalize_quaternion(quat):
    quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat.reshape(-1, 4)
    for i in range(quat.shape[0]):
        if quat[i, -1] < 0:
            quat[i] = -quat[i]
    return quat


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = R.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc

# -----------------------------------------------------------------------------
# HEIGHTMAP UTILS
# -----------------------------------------------------------------------------

def get_fused_heightmap(colors, points, bounds, pix_size):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = [], []
    for p,i in zip(points, colors):
        h, c_map = get_heightmap(p, i, bounds, pix_size)
        heightmaps.append(h)
        colormaps.append(c_map)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.
    return cmap, hmap

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
  
    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.
  
    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap

def parse_state(filename):
    units = filename.split('-')
    if 'pickup_object' in filename:
        init_state = 0
        goal_state = int(units[-3]) / 40
    elif 'reorient_object' in filename:
        init_state = 0.5
        goal_state = int(filename.split(')')[0].split('_')[-1]) / 180
    elif 'pour_water' in filename:
        init_state = 0
        goal_state = (100 - int(units[-3])) / 100
    elif 'transfer_water' in filename:
        init_state = 0
        goal_state = int(units[-3]) / 100
    else:
        init_state = float(units[-4])
        goal_state = float(units[-3])

    return init_state, goal_state

def create_lang_encoder(cfg, device):
    lang_encoder = 'clip'
    if lang_encoder == 'clip':
        return CLIP_encoder(device)
    # elif cfg.lang_encoder == 't5':
    #     return T5_encoder(cfg.t5_cfg, device)
    # elif cfg.lang_encoder == 'roberta':
    #     return RoBERTa(cfg.roberta_cfg, device)
    # elif cfg.lang_encoder == 'none':
    #     return None
    else:
        raise ValueError('Language encoder key not supported')

def prepare_batch(dataset, batch_data, lang_embed_cache=None, device='cuda:0'):
        if lang_embed_cache is None:
            lang_embed_cache = dataset.lang_embed_cache
        bs = batch_data['input_images'].shape[0]

        current_states = batch_data['current_state'].cpu().numpy()
        goal_states = batch_data['goal_state'].cpu().numpy()
        current_states = np.array(current_states).reshape(bs, 1)
        goal_states = np.array(goal_states).reshape(bs, 1)
        states = np.concatenate([current_states, goal_states], axis=1)   # [bs, 2]

        target_points = batch_data['target_points'].cpu().numpy()
        trans_action_coords = target_points[:, :3]
        voxel_size = 120
        offset_bound = [-0.63, 0, -0.63, 0.63, 1.26, 0.63]
        trans_action_indices = point_to_voxel_index(trans_action_coords, voxel_size, offset_bound)

        gripper_open = batch_data['target_gripper'].cpu().numpy().reshape(bs, 1)
        rot_action_quat = target_points[:, 3:]
        rot_action_quat = normalize_quaternion(rot_action_quat)
        rotation_resolution = 5
        rot_action_indices = quaternion_to_discrete_euler(rot_action_quat, rotation_resolution)
        rot_grip_action_indices = np.concatenate([rot_action_indices, gripper_open], axis=-1)

        language_instructions = batch_data['language']
        lang_goal_embs = lang_embed_cache.get_lang_embed(language_instructions)

        inp = {}
        inp.update({
            'trans_action_indicies': torch.from_numpy(trans_action_indices).long(),
            'rot_grip_action_indicies': torch.from_numpy(rot_grip_action_indices).long(),
            'states': torch.from_numpy(states),
            'lang_goal_embs': lang_goal_embs.to(device, dtype=torch.float),
        })

        for k in inp.keys():
            batch_data[k] = inp[k].to(device)
        

        return batch_data

class ArnoldDataset(Dataset):
    def __init__(self, data_path, task, cfg, obs_type='rgb', offset_bound=np.array([-0.63, 0, -0.63, 0.63, 1.26, 0.63]), canonical=False):
        """
        Dataset structure: {
            object_id: {
                'act1': List,
                'act2': List
            }
        }
        """
        super().__init__()
        self.data_path = data_path
        self.task = task
        self.pixel_size = 5.625e-3
        self.obs_type = obs_type
        self.task_offset = offset_bound.reshape(2, 3).transpose(1, 0)
        self.sample_weights = [0.2, 0.8]
        self.obj_ids = []
        self.episode_dict = {}
        self.lang_encoder = create_lang_encoder(cfg, DEVICE)
        self.lang_embed_cache = InstructionEmbedding(self.lang_encoder)
        self.canonical = canonical
        self._load_keyframes()
        self.cfg = cfg
    
    def prepare_batch(self, batch_data, cfg=None, lang_embed_cache=None, device=DEVICE):
        if lang_embed_cache is None:
            lang_embed_cache = self.lang_embed_cache
        if cfg is None:
            cfg = self.cfg

        obs_dict = {}
        language_instructions = []
        target_points = []
        gripper_open = []
        low_dim_state = []
        current_states = []
        goal_states = []
        ext_matrices = []
        intr_matrices = []
        input_rgb_images = []
        target_rgb_images = []
        input_camera_positions = []
        target_camera_positions = []
        input_rays = []
        target_rays = []
        keyframes = []
        trans_point_2d = []

        for data in batch_data:
            for k, v in data['obs_dict'].items():
                if k not in obs_dict:
                    obs_dict[k] = [v]
                else:
                    obs_dict[k].append(v)
        
            target_points.append(data['target_points'])
            gripper_open.append(data['target_gripper'])
            language_instructions.append(data['language'])
            low_dim_state.append(data['low_dim_state'])
            current_states.append(data['current_state'])
            goal_states.append(data['goal_state'])
            ext_matrices.append(data['ext_matrices'])
            intr_matrices.append(data['intr_matrices'])
            input_rgb_images.append(data['input_images'])
            target_rgb_images.append(data['target_images'])
            input_camera_positions.append(data['input_camera_pos'])
            target_camera_positions.append(data['target_camera_pos'])
            input_rays.append(data['input_rays'])
            target_rays.append(data['target_rays'])
            keyframes.append(data['keyframe'])
            trans_point_2d.append(data['trans_point_2d'])


        for k, v in obs_dict.items():
            v = np.stack(v, axis=0)
            obs_dict[k] = v.transpose(0, 3, 1, 2)   # peract requires input as [C, H, W]
        ext_matrices = np.stack(ext_matrices, axis=0)
        intr_matrices = np.stack(intr_matrices, axis=0)
        input_rgb_images = np.stack(input_rgb_images, axis=0)
        target_rgb_images = np.stack(target_rgb_images, axis=0)
        input_rgb_images = input_rgb_images.transpose(0, 1, 4, 2, 3)
        target_rgb_images = target_rgb_images.transpose(0, 1, 4, 2, 3)
        input_camera_positions = np.stack(input_camera_positions, axis=0)
        target_camera_positions = np.stack(target_camera_positions, axis=0)
        input_rays = np.stack(input_rays, axis=0)
        target_rays = np.stack(target_rays, axis=0)
        keyframes = np.stack(keyframes, axis=0)
        trans_point_2d = np.stack(trans_point_2d, axis=0)

        bs = len(language_instructions)
        target_points = np.stack(target_points, axis=0)
        gripper_open = np.array(gripper_open).reshape(bs, 1)
        low_dim_state = np.stack(low_dim_state, axis=0)

        current_states = np.array(current_states).reshape(bs, 1)
        goal_states = np.array(goal_states).reshape(bs, 1)
        states = np.concatenate([current_states, goal_states], axis=1)   # [bs, 2]

        trans_action_coords = target_points[:, :3]
        voxel_size = 120
        offset_bound = [-0.63, 0, -0.63, 0.63, 1.26, 0.63]
        trans_action_indices = point_to_voxel_index(trans_action_coords, voxel_size, offset_bound)

        rot_action_quat = target_points[:, 3:]
        rot_action_quat = normalize_quaternion(rot_action_quat)
        rotation_resolution = 5
        rot_action_indices = quaternion_to_discrete_euler(rot_action_quat, rotation_resolution)
        rot_grip_action_indices = np.concatenate([rot_action_indices, gripper_open], axis=-1)

        lang_goal_embs = lang_embed_cache.get_lang_embed(language_instructions)

        inp = {}
        inp.update(obs_dict)
        inp.update({
            'input_images': input_rgb_images,
            'target_images': target_rgb_images,
            'ext_matrices': ext_matrices,
            'intr_matrices': intr_matrices,
            'input_camera_pos': input_camera_positions,
            'target_camera_pos': target_camera_positions,
            'input_rays': input_rays,
            'target_rays': target_rays,
            'keyframes': keyframes,
            'trans_action_indicies': trans_action_indices,
            'rot_grip_action_indicies': rot_grip_action_indices,
            'states': states,
            'lang_goal_embs': lang_goal_embs,
            'low_dim_state': low_dim_state,
            'trans_point_2d': trans_point_2d,
        })

        for k, v in inp.items():
            if v is not None:
                if not isinstance(v, torch.Tensor):
                    v = torch.from_numpy(v)
                inp[k] = v.to(device)
        return inp
    
    def _load_keyframes(self):
        """
        Generally, there are 4 frames in each demonstration:
            0: initial state
            1: pre-grasping state
            2: grasping state
            3: final goal state
        In this setting, two samples are extracted for training:
            - observation at initial state (0) as visual input, gripper state at grasping state (2) as action label
            - observation at grasping state (2) as visual input, gripper state at final goal state (3) as action label
        For tasks involving water, the difference is as follows:
            3: to the position before tilting
            4: to the orientation before reverting the cup
            5: final upwards state
            - combination of position at frame 3 and orientation at frame 4 as action label
        All template actions are in world frame.
        """
        for fname in tqdm(os.listdir(self.data_path)):
            if fname.endswith('npz'):
                obj_id = fname.split('-')[2]
                if obj_id not in self.episode_dict:
                    self.obj_ids.append(obj_id)
                    self.episode_dict.update({
                        obj_id: {
                            'act1': [],
                            'act2': []
                        }
                    })

                init_state, goal_state = parse_state(fname)

                gt_frames = np.load(os.path.join(self.data_path, fname), allow_pickle=True)['gt']
                language_instructions = gt_frames[0]['instruction']

                # pick phase
                step = gt_frames[0].copy()
                robot_base_pos = step['robot_base'][0] / 100
                robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
                robot_forward_direction[1] = 0   # height
                robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
                bound_center = robot_base_pos + robot_forward_direction
                print(f'ROBOT BASE: {robot_base_pos}')
                print(f'BOUND CENTER: {bound_center}')

                cmap, hmap, obs_dict = self.get_step_obs(step, self.task_offset[[0, 2, 1]], self.pixel_size, type=self.obs_type)
                rgb_images, camera_locations, camera_rays, ext_matrices, int_matrices, _ = get_data_from_cameras(step['images'])
                
                # convert to meters
                # for i in range(len(int_matrices)):
                #     int_matrices[i] /= 100
            
                hmap = np.tile(hmap[..., None], (1,1,3))
                img = np.concatenate([cmap, hmap], axis=-1)

                obj_pos = gt_frames[2]['position_rotation_world'][0] / 100
                obj_pos = obj_pos # - bound_center

                act_pos = gt_frames[2]['position_rotation_world'][0].copy()
                act_rot = gt_frames[2]['position_rotation_world'][1].copy()
                act_pos /= 100
                act_rot = act_rot[[1,2,3,0]]   # wxyz to xyzw
                target_points = self.get_act_label_from_abs(pos_abs=act_pos, rot_abs=act_rot)
                target_points[:3] = target_points[:3]  # - bound_center
                print(f'TARGET POINTS: {target_points}')

                gripper_open = 1
                gripper_joint_positions = step['gripper_joint_positions'] / 100
                gripper_joint_positions = np.clip(gripper_joint_positions, 0, 0.04)
                timestep = 0
                low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep])
                target_rays = camera_rays[5:].reshape(-1, 3)
                target_camera_pos = einops.repeat(camera_locations[5:], 'm n -> m k n', k=target_rays.shape[0] // camera_locations[5:].shape[0]).reshape(-1,3)
                trans_pose = torch.FloatTensor(target_points[:3])
        
                point_2d = []
                pose = np.append(target_points[:3], 1)
                for i in range(5, 10):
                    point = np.linalg.inv(ext_matrices[i]) @ pose
                    pt = int_matrices[i] @ point[:-1]
                    pt = np.rint(pt[:-1] / pt[-1])
                    point_2d.append(pt)
                
                trans_point_2d = torch.LongTensor(np.array(point_2d))
                
                input_rays = camera_rays[:3]
                input_camera_pos = camera_locations[:3]

                if self.canonical:
                    canonical_extrinsic = torch.FloatTensor(np.linalg.inv(ext_matrices[5]))
                    input_rays = torch.FloatTensor(input_rays)
                    input_camera_pos = torch.FloatTensor(input_camera_pos)
                    target_rays = torch.FloatTensor(target_rays)
                    target_camera_pos = torch.FloatTensor(target_camera_pos)

                    input_rays = transform_points_torch(input_rays, canonical_extrinsic, translate=False)
                    input_camera_pos = transform_points_torch(input_camera_pos, canonical_extrinsic)
                    target_rays = transform_points_torch(target_rays, canonical_extrinsic, translate=False)
                    target_camera_pos = transform_points_torch(target_camera_pos, canonical_extrinsic)

                robot_pose = np.eye(4, 4)
                robot_pose[:3, :3] = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()
                robot_pose[:3, -1] = robot_base_pos

                new_ext = np.zeros_like(ext_matrices)

                for i, ext in enumerate(ext_matrices):
                    new_ext[i] = robot_pose @ np.linalg.inv(ext_matrices[i].T)
                
                ext_matrices = new_ext

                episode_dict1 = {
                    "rgb_images": rgb_images,
                    "input_images": rgb_images[:3],
                    "target_images": rgb_images[5:],
                    "target_pixels": rgb_images[5:].reshape(-1, 3),
                    "input_camera_pos": input_camera_pos,
                    "target_camera_pos": target_camera_pos,
                    "input_rays": input_rays,
                    "target_rays": target_rays,
                    "ext_matrices": ext_matrices,
                    "intr_matrices": int_matrices,
                    "obs_dict": obs_dict,   # { {camera_name}_{rgb/point_cloud}: [H, W, 3] }
                    "attention_points": obj_pos,   # [3,]
                    "target_points": target_points,   # [6,]
                    "target_gripper": gripper_open,   # binary
                    "low_dim_state": low_dim_state,   # [grip_open, left_finger, right_finger, timestep]
                    "language": language_instructions,   # str
                    "current_state": init_state,   # scalar
                    "goal_state": goal_state,   # scalar
                    "bounds": self.task_offset,   # [3, 2]
                    "pixel_size": self.pixel_size,   # scalar
                    "keyframe": timestep,
                    "trans_point_2d": trans_point_2d,
                }

                self.episode_dict[obj_id]['act1'].append(episode_dict1)

                # place phase
                step = gt_frames[2].copy()
                robot_base_pos = step['robot_base'][0] / 100
                robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
                robot_forward_direction[1] = 0   # height
                robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
                bound_center = robot_base_pos + robot_forward_direction

                rgb_images, camera_locations, camera_rays, ext_matrices, int_matrices, _ = get_data_from_cameras(step['images'])
                # convert to meters
                # for i in range(len(int_matrices)):
                #     int_matrices[i] /= 100

                cmap, hmap, obs_dict = self.get_step_obs(step, self.task_offset[[0, 2, 1]], self.pixel_size, type=self.obs_type)
                hmap = np.tile(hmap[..., None], (1,1,3))
                img = np.concatenate([cmap, hmap], axis=-1)

                obj_pos = step['position_rotation_world'][0] / 100 - bound_center

                act_pos = gt_frames[3]['position_rotation_world'][0].copy()
                if self.task in ['pour_water', 'transfer_water']:
                    # water, compose actions of two frames
                    act_rot = gt_frames[4]['position_rotation_world'][1].copy()
                else:
                    # default
                    act_rot = gt_frames[3]['position_rotation_world'][1].copy()

                act_pos /= 100
                act_rot = act_rot[[1,2,3,0]]   # wxyz to xyzw
                target_points = self.get_act_label_from_abs(pos_abs=act_pos, rot_abs=act_rot)
                target_points[:3] = target_points[:3] # - bound_center

                gripper_open = 0
                gripper_joint_positions = step['gripper_joint_positions'] / 100
                gripper_joint_positions = np.clip(gripper_joint_positions, 0, 0.04)
                timestep = 1
                low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep])
                target_rays = camera_rays[5:].reshape(-1, 3)
                target_camera_pos = einops.repeat(camera_locations[5:], 'm n -> m k n', k=target_rays.shape[0] // camera_locations[5:].shape[0]).reshape(-1,3)
                trans_pose = torch.FloatTensor(target_points[:3])
        
                point_2d = []
                pose = np.append(target_points[:3], 1)
                for i in range(5, 10):
                    point = np.linalg.inv(ext_matrices[i]) @ pose
                    pt = int_matrices[i] @ point[:-1]
                    pt = np.rint(pt[:-1] / pt[-1])
                    point_2d.append(pt)
                
                trans_point_2d = torch.LongTensor(np.array(point_2d))
                
                input_rays = camera_rays[:3]
                input_camera_pos = camera_locations[:3]

                if self.canonical:
                    canonical_extrinsic = torch.FloatTensor(np.linalg.inv(ext_matrices[5]))
                    input_rays = torch.FloatTensor(input_rays)
                    input_camera_pos = torch.FloatTensor(input_camera_pos)
                    target_rays = torch.FloatTensor(target_rays)
                    target_camera_pos = torch.FloatTensor(target_camera_pos)
                    
                    input_rays = transform_points_torch(input_rays, canonical_extrinsic, translate=False)
                    input_camera_pos = transform_points_torch(input_camera_pos, canonical_extrinsic)
                    target_rays = transform_points_torch(target_rays, canonical_extrinsic, translate=False)
                    target_camera_pos = transform_points_torch(target_camera_pos, canonical_extrinsic)


                robot_pose = np.eye(4, 4)
                robot_pose[:3, :3] = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()
                robot_pose[:3, -1] = robot_base_pos

                new_ext = np.zeros_like(ext_matrices)

                for i, ext in enumerate(ext_matrices):
                    new_ext[i] = robot_pose @ np.linalg.inv(ext_matrices[i].T)
                
                ext_matrices = new_ext

                episode_dict2 = {
                    "rgb_images": rgb_images,
                    "input_images": rgb_images[:3],
                    "target_images": rgb_images[5:],
                    "target_pixels": rgb_images[5:].reshape(-1, 3),
                    "input_camera_pos": input_camera_pos,
                    "target_camera_pos": target_camera_pos,
                    "input_rays": input_rays,
                    "target_rays": target_rays,
                    "ext_matrices": ext_matrices,
                    "intr_matrices": int_matrices,
                    "obs_dict": obs_dict,   # { {camera_name}_{rgb/point_cloud}: [H, W, 3] }
                    "attention_points": obj_pos,   # [3,]
                    "target_points": target_points,   # [6,]
                    "target_gripper": gripper_open,   # binary
                    "low_dim_state": low_dim_state,   # [grip_open, left_finger, right_finger, timestep]
                    "language": language_instructions,   # str
                    "current_state": init_state,   # scalar
                    "goal_state": goal_state,   # scalar
                    "bounds": self.task_offset,   # [3, 2]
                    "pixel_size": self.pixel_size,   # scalar
                    "keyframe": timestep,
                    "trans_point_2d": trans_point_2d,
                }

                self.episode_dict[obj_id]['act2'].append(episode_dict2)

        print(f'Loaded {[len(v["act1"]) for k, v in self.episode_dict.items()]} demos')

    def __len__(self):
        num_demos = 0
        for k, v in self.episode_dict.items():
            num_demos += len(v['act1'])
        return num_demos

    def __getitem__(self, index):
        obj_demos = [len(v['act1']) for k, v in self.episode_dict.items()]
        interval_upper = np.cumsum(obj_demos)
        interval_lower = np.array([0, *interval_upper[:-1]])

        obj_idx = ((index>=interval_lower) * (index<interval_upper)).tolist().index(True)
        demo_idx = index - interval_lower[obj_idx]
        obj_idx = self.obj_ids[obj_idx]

        act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)[0]
        return self.episode_dict[obj_idx][f'act{act_idx}'][demo_idx]

        # sample
        # obj_idx = random.choice(list(self.episode_dict.keys()))
        # act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)[0]
        # return random.choice(self.episode_dict[obj_idx][f'act{act_idx}'])

    def sample(self, batch_size=None, cfg=None):
        if cfg is None:
            cfg = self.cfg
        if batch_size is None:
            batch_size = 1
            
        samples = []
        sampled_idx = []
        while len(samples) < batch_size:
            obj_idx = random.choice(self.obj_ids)
            act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)[0]
            demo_idx = np.random.randint(len(self.episode_dict[obj_idx][f'act{act_idx}']), size=1)[0]
            obj_act_demo_tuple = (obj_idx, act_idx, demo_idx)
            if obj_act_demo_tuple not in sampled_idx:
                samples.append(self.episode_dict[obj_idx][f'act{act_idx}'][demo_idx])
                sampled_idx.append(obj_act_demo_tuple)

        return self.prepare_batch(samples, cfg)

    def get_step_obs(self, step, bounds, pixel_size, type='rgb'):
        # bounds: [3, 2], xyz (z-up)
        imgs = step['images']
        colors = []
        pcds = []
        obs_dict = {}
        for camera_name, camera_obs in zip(CAMERAS, imgs):
            camera = camera_obs['camera']

            if type == 'rgb':
                color = camera_obs['rgb'][:, :, :3]
            elif type == 'mask':
                color = camera_obs['semanticSegmentation'][:,:,np.newaxis].repeat(3,-1) * 50
            else:
                raise ValueError('observation type should be either rgb or mask')
            colors.append(color)

            depth = np.clip(camera_obs['depthLinear'], 0, 10)

            point_cloud = create_pcd_hardcode(camera, depth, cm_to_m=True)
            # here point_cloud is y-up
            robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
            robot_forward_direction[1] = 0
            robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
            bound_center = step['robot_base'][0] / 100 + robot_forward_direction
            point_cloud = point_cloud # - bound_center

            pcds.append(point_cloud[:, :, [0, 2, 1]])   # pcds is for cliport6d, which requires z-up

            obs_dict.update({
                f'{camera_name}_rgb': color,
                f'{camera_name}_point_cloud': point_cloud
            })   # obs_dict is for peract, which requires y-up

        # to fuse map, pcds and bounds are supposed to be xyz (z-up)
        cmap, hmap = get_fused_heightmap(colors, pcds, bounds, pixel_size)
        return cmap, hmap, obs_dict

    def get_act_label_from_rel(self, pos_rel, rot_rel, robot_base):
        robot_pos, robot_rot = robot_base
        robot_rot = R.from_quat(robot_rot).as_matrix()
        pos_world, rot_world = get_pose_world(pos_rel, rot_rel, robot_pos, robot_rot)
        if rot_world is None:
            return pos_world
        else:
            rot_world = R.from_matrix(rot_world).as_quat()
            return np.concatenate([pos_world, rot_world])

    def get_act_label_from_abs(self, pos_abs, rot_abs):
        if rot_abs is None:
            return pos_abs
        else:
            return np.concatenate([pos_abs, rot_abs])

class ArnoldMultiTaskDataset(Dataset):
    task_list = [
        'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
        'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
    ]
    def __init__(self, data_root, obs_type='rgb'):
        """
        Dataset structure: [
            {
                object_id: {
                    'act1': List,
                    'act2': List
                }
            }
        ]
        """
        super().__init__()
        self.data_root = data_root
        self.obs_type = obs_type
        self.task_dict = {}
        for task in self.task_list:
            print(f'Create dataset for {task}')
            self.task_dict[task] = ArnoldDataset(os.path.join(data_root, task, 'train'), task, obs_type)

    def __len__(self):
        num_demos = 0
        for k, v in self.task_dict.items():
            num_demos += len(v)
        return num_demos
    
    def __getitem__(self, index):
        demos_per_task = [len(v) for k, v in self.task_dict.items()]
        interval_upper = np.cumsum(demos_per_task)
        interval_lower = np.array([0, *interval_upper[:-1]])

        task_idx = ((index>=interval_lower) * (index<interval_upper)).tolist().index(True)
        demo_idx = index - interval_lower[task_idx]
        return self.task_dict[self.task_list[task_idx]][demo_idx]
    
    def sample(self, batch_size):
        samples = []
        while len(samples) < batch_size:
            task = random.choice(self.task_list)
            samples.append(self.task_dict[task].sample(1)[0])
        
        return samples


class InstructionEmbedding():
    def __init__(self, lang_encoder):
        self.cache = {}
        self.lang_encoder = lang_encoder
    
    def get_lang_embed(self, instructions):
        if self.lang_encoder is None:
            return None
        
        if isinstance(instructions, str):
            # a single sentence
            instructions = [instructions]
        
        lang_embeds = []
        for sen in instructions:
            if sen not in self.cache:
                sen_embed = self.lang_encoder.encode_text([sen])
                sen_embed = sen_embed[0]
                self.cache.update({sen: sen_embed})
            
            lang_embeds.append(self.cache[sen])
        
        lang_embeds = torch.stack(lang_embeds, dim=0)
        return lang_embeds