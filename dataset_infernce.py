from dataset import ArnoldDataset
import hydra
import os

@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    dataset = ArnoldDataset('/home/chemrobot/Documents/RichardHanxu2023/arnold/data/transfer_water/train', 'transfer_water', cfg)
    sample = dataset.sample(cfg.batch_size, cfg)
    print(sample.keys())
    print(sample['camera_rays'].shape)
    print(sample['keyframes'].shape)
    print(sample['rgb_images'].shape)
    print(sample['ext_matrices'].shape)
    print(sample['intr_matrices'].shape)
    print(sample['camera_positions'].shape)
    
if __name__ == '__main__':
    main()