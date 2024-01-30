# This file contains the code for synthetic cloud microphysics dataset loaders for VIP-CT and ProbCT.
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

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../Data')

ALL_DATASETS = ("BOMEX_10cams_polarization")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


def get_cloud_microphysics_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'BOMEX_10cams_polarization':
        data_root = '/wdata/yaelsc/Roi_ICCP22/'
        image_size = [123, 123]

    if dataset_name == 'BOMEX_10cams_polarization':
        print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
        data_train_paths = [f for f in glob.glob(os.path.join(data_root, "CloudCT10sat_polarized_LWC_NOISE/*.pkl"))]

    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(data_train_paths)

    data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    rand_cam = cfg.data.rand_cam
    train_dataset = MicrophysicsCloudDataset(
            data_train_paths,
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
    )
    if dataset_name == 'BOMEX_10cams_polarization':
        val_paths =  [f for f in glob.glob(os.path.join(data_root, "CloudCT10sat_polarized_LWC_NOISE_TEST/*.pkl"))]
    else:
        NotImplementedError()
    val_len = cfg.data.n_val if cfg.data.n_val>0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = MicrophysicsCloudDataset(val_paths, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,   dataset_name = dataset_name)
    return train_dataset, val_dataset


class MicrophysicsCloudDataset(Dataset):
    def __init__(self, cloud_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None, mean=0, std=1, dataset_name=''):
        self.cloud_dir = cloud_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.rand_cam = rand_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]
        if 'TEST' in cloud_path:
            data_root = os.path.join(DEFAULT_DATA_ROOT, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras', 'test')
        else:
            data_root = os.path.join(DEFAULT_DATA_ROOT, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras', 'train')
        image_index = cloud_path.split('satellites_images_')[-1].split('.pkl')[0]
        projection_path = os.path.join(data_root, f"cloud_results_{image_index}.pkl")
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        images = np.concatenate((data['images'], data['DoLPs'][:, None], data['AoLPs'][:, None]), 1)
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        if 'varying' in self.dataset_name:
            index = torch.randperm(10)[0]
            cam_i = torch.arange(index,100,10)
            mask = mask[index] if mask is not None else None
        else:
            cam_i = torch.arange(self.n_cam)
        images = images[cam_i]
        images -= np.array(self.mean).reshape((1,5,1,1))
        images /= np.array(self.std).reshape((1,5,1,1))

        microphysics = np.array([data['lwc'],data['reff'],data['veff']])
        microphysics = microphysics[:,1:-1,1:-1,:-1]
        mask = mask[1:-1,1:-1,:-1]
        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape[1:] for image in images]
        with open(projection_path, 'rb') as f:
            data = pickle.load(f)

        grid = data['grid']
        camera_center = data['cameras_pos'][cam_i]
        projection_matrix = data['cameras_P'][cam_i]

        return images, microphysics, grid, image_sizes, projection_matrix, camera_center, mask
