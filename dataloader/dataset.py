# This file contains the code for synthetic cloud dataset loaders for VIP-CT and ProbCT.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# authors. The authors are not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.


import os, glob
from typing import Tuple
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import socket
import random
import scipy.io as sio
from .noise import SatelliteNoise
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../Data')

ALL_DATASETS = ("Toy_10cameras_20m","Toy2_10cameras_20m","Toy3_10cameras_20m","BOMEX_CASS_10cameras_20m", "CASS_10cameras_20m", "CASS_10cameras_50m", "BOMEX_10cameras_20m",
                "BOMEX_10cameras_50m", "BOMEX_32cameras_20m", "BOMEX_32cameras_50m", "BOMEX_10cameras_20m_varying_S", "BOMEX_10cameras_20m_varying_M",
                "BOMEX_10cameras_20m_varying_L", "BOMEX_10cameras_20m_varying_XL",
                "subset_of_seven_clouds",
                "BOMEX_50CCN_10cameras_20m",
                "BOMEX_50CCN_10cameras_20m_pseudo_label_no_noise",
                "BOMEX_50CCN_10cameras_20m_pseudo_label_noise",
                "BOMEX_50CCN_aux_10cameras_20m",
                'CASS_600CCN_roiprocess_10cameras_20m',
                "HAWAII_2000CCN_10cameras_20m",
                "DYCOMS_RF02_500CCN_10cameras_20m",
                "DYCOMS_RF02_50CCN_10cameras_20m",
                "BOMEX_5000CCN_new_10cameras_20m",
                )


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


def get_cloud_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        cfg: The config file with the name of the dataset to load.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
    """
    dataset_names = cfg.data.dataset_name
    if isinstance(dataset_names,str):
        dataset_names = [dataset_names]
    else:
        dataset_names = list(dataset_names)
    train_paths = []
    val_paths = []
    for dataset_name in dataset_names:
        if dataset_name not in ALL_DATASETS:
            raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

        if dataset_name == 'CASS_10cameras_20m':
            data_root = os.path.join(data_root, 'CASS_50m_256x256x139_600CCN/10cameras_50m')
            image_size = [236, 236]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_CASS_10cameras_20m':
            data_root_cass = os.path.join(data_root, 'CASS_50m_256x256x139_600CCN/10cameras_20m')
            data_root_bomex = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_20m')
            cfg.data.image_size = [236, 236]
        elif dataset_name == 'CASS_10cameras_50m':
            data_root = os.path.join(data_root, 'CASS_50m_256x256x139_600CCN/10cameras_50m')
            image_size = [96, 96]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_10cameras_20m':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_10cameras_20m_varying_S':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_20m_varying_S')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_10cameras_20m_varying_M':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_20m_varying_M')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_10cameras_20m_varying_L':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_20m_varying_L')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_10cameras_20m_varying_XL':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_20m_varying_XL')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_10cameras_50m':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_50m')
            image_size = [48, 48]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_32cameras_50m':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '32cameras_50m')
            image_size = [48, 48]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_32cameras':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '32cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'subset_of_seven_clouds':
            data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_50m')
            image_size = [48, 48]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_50CCN_10cameras_20m':
            data_root = os.path.join(data_root, 'BOMEX_128x128x100_50CCN_50m_micro_256', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_50CCN_10cameras_20m_pseudo_label_no_noise':
            data_root = os.path.join(data_root, 'BOMEX_128x128x100_50CCN_50m_micro_256', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_50CCN_10cameras_20m_pseudo_label_noise':
            data_root = os.path.join(data_root, 'BOMEX_128x128x100_50CCN_50m_micro_256', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_50CCN_aux_10cameras_20m':
            data_root = os.path.join(data_root, 'BOMEX_128x128x100_50CCN_50m_micro_256_aux4', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'BOMEX_5000CCN_new_10cameras_20m':
            data_root = os.path.join(data_root, 'BOMEX_128x128x100_5000CCN_50m_micro_256', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
            data_root = os.path.join(data_root, 'CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'HAWAII_2000CCN_10cameras_20m':
            data_root = os.path.join(data_root, 'HAWAII_2000CCN_32x32x64_50m', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'DYCOMS_RF02_50CCN_10cameras_20m':
            data_root = os.path.join(data_root, 'DYCOMS_RF02_50CCN_64x64x159_50m', '10cameras_20m')
            image_size = [236, 236]
            cfg.data.image_size = image_size
        elif dataset_name == 'DYCOMS_RF02_500CCN_10cameras_20m':
            data_root = os.path.join(data_root, 'DYCOMS_RF02_500CCN_64x64x159_50m', '10cameras_20m')
            image_size = [236, 236]
            cfg.data.image_size = image_size

        elif dataset_name == 'Toy_10cameras_20m':
            data_root = os.path.join(data_root, 'Toy', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'Toy2_10cameras_20m':
            data_root = os.path.join(data_root, 'Toy2', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        elif dataset_name == 'Toy3_10cameras_20m':
            data_root = os.path.join(data_root, 'Toy_single_voxel_clouds', '10cameras_20m')
            image_size = [116, 116]
            cfg.data.image_size = image_size
        else:
            FileNotFoundError()

        if not dataset_name == 'BOMEX_CASS_10cameras_20m':
            print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
            if 'pseudo_label_noise' in dataset_name:
                data_train_path = [f for f in glob.glob(os.path.join(data_root, "pseudo_train_noise/cloud*.pkl"))]
            elif 'pseudo_label_no_noise' in dataset_name:
                data_train_path = [f for f in glob.glob(os.path.join(data_root, "pseudo_train_no_noise/cloud*.pkl"))]
            else:
                data_train_path = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]
            train_len = cfg.data.n_training if cfg.data.n_training > 0 else len(data_train_path)
            train_paths += data_train_path[:train_len]
        else:
            print(f"Loading dataset {dataset_name}...")
            data_train_path1 = [f for f in glob.glob(os.path.join(data_root_cass, "train/cloud*.pkl"))]
            train_len = cfg.data.n_training if cfg.data.n_training > 0 else len(data_train_path1)
            data_train_paths = data_train_path1[:train_len]
            data_train_path2 = [f for f in glob.glob(os.path.join(data_root_bomex, "train/cloud*.pkl"))]
            train_len = cfg.data.n_training if cfg.data.n_training > 0 else len(data_train_path2)
            data_train_paths += data_train_path1[:train_len]

        if not (dataset_name == 'BOMEX_CASS_10cameras_20m' or dataset_name == 'subset_of_seven_clouds'):
            val_path = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
            val_len = cfg.data.n_val if cfg.data.n_val > 0 else len(val_path)
            val_paths += val_path[:val_len]
        elif dataset_name == 'subset_of_seven_clouds':
            val_path = [f for f in glob.glob(os.path.join(data_root, "subset_of_seven_clouds/cloud*.pkl"))]
            val_paths += val_path
        else:
            val_path = [f for f in glob.glob(os.path.join(data_root_cass, "test/cloud*.pkl"))]
            val_len = cfg.data.n_val if cfg.data.n_val > 0 else len(val_path)
            val_paths += val_path[:val_len]

            val_path = [f for f in glob.glob(os.path.join(data_root_bomex, "test/cloud*.pkl"))]
            val_len = cfg.data.n_val if cfg.data.n_val > 0 else len(val_path)
            val_paths += val_path[:val_len]




        # val_paths = val_paths[:val_len]
    random.shuffle(train_paths)
    random.shuffle(val_paths)

    # data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    rand_cam = cfg.data.rand_cam

    train_dataset = CloudDataset(
            train_paths,
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
    dataset_name = dataset_names,
        noise=cfg.data.noise,
        full_well_val= cfg.data.full_well_val if hasattr(cfg.data, 'full_well_val') else None
    )

    val_dataset = CloudDataset(val_paths, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,   dataset_name = dataset_name,
        noise=cfg.data.noise,
        full_well_val=cfg.data.full_well_val if hasattr(cfg.data, 'full_well_val') else None
)

    return train_dataset, val_dataset


class CloudDataset(Dataset):
    def __init__(self, cloud_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None, mean=0, std=1, dataset_name='',
                 fix_grid=False, noise=False, full_well_val=None):
        self.cloud_dir = cloud_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.rand_cam = rand_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.fix_grid = fix_grid
        self.noise = SatelliteNoise(full_well_val=full_well_val) if noise else None

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        # if hasattr(data,'observed_images'):
        #     images = data['observed_images']
        # else:
        #     images = data['images']
        images = data['images']
        if self.transform:
            images = self.transform(self.transform)
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        elif self.mask_type == 'space_carving_0.9':
            mask = data['mask0.9']
        elif self.mask_type == 'toy3':
            mask = np.zeros((32*32*64)).astype(bool)
            mask[data['index']] = True
            mask = mask.reshape((32,32,64))
            # images = images[:,:][None]
        if 'varying' in self.dataset_name:
            # randomly sample a perturbation out of the tenth simulated data
            index = torch.randperm(10)[0]
            cam_i = torch.arange(index,100,10)
            mask = mask[index] if mask is not None else None
        else:
            cam_i = torch.arange(self.n_cam)
        images = images[cam_i]
        if len(images.shape)==2:
            images = images[None]
        if self.noise is not None:
            images = self.noise.convert_radiance_to_graylevel(images)

        images -= self.mean
        images /= self.std

        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape for image in images]
        extinction = data['ext']
        grid = data['grid'] #there is an issue with some grids. make sure that the grids starts from (0,0,0)
        if self.fix_grid:
            grid[0] -= grid[0][0]
            grid[1] -= grid[1][0]
            grid[2] -= grid[2][0]
        camera_center = data['cameras_pos'][cam_i]
        projection_matrix = data['cameras_P'][cam_i]

        return images, extinction, grid, image_sizes, projection_matrix, camera_center, mask, cloud_path
