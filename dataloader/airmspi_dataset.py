# This file contains the code for real-world AirMSPI cloud dataset loaders for VIP-CT.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen, Vadim Holodovsky and Yoav. Y. Schechner, "Variable Imaging Projection Cloud Scattering Tomography",
# Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
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

DEFAULT_DATA_ROOT = '/wdata/roironen/Data'

def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch

ALL_DATASETS_AIRMSPI = ("32N123W_experiment_all_clouds",'32N123W_experiment_234_clouds',
    "AirMSPI_BOMEX_50CCN_9cams","AirMSPI_BOMEX_aux_9cams", "32N123W_experiment_cloud1", "32N123W_experiment_cloud2", "18S8E_experiment"
)
#
# def get_airmspi_datasets(
#     cfg,
#     data_root: str = DEFAULT_DATA_ROOT,
# ) -> Tuple[Dataset, Dataset, int]:
#     """
#     Obtains the training and validation dataset object for a dataset specified
#     with the `dataset_name` argument.
#
#     Args:
#         dataset_name: The name of the dataset to load.
#         image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
#         data_root: The root folder at which the data is stored.
#
#     Returns:
#         train_dataset: The training dataset object.
#         val_dataset: The validation dataset object.
#         test_dataset: The testing dataset object.
#     """
#     dataset_name = cfg.data.dataset_name
#
#     if dataset_name not in ALL_DATASETS_AIRMSPI:
#         raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")
#
#     if dataset_name == 'BOMEX_9cams':
#         data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
#         image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/AIRMSPI_IMAGES_LWC_LOW_SC_NOISY_PROJ/'
#         mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#         image_size = [350, 350]
#     elif dataset_name == 'BOMEX_9cams_varying':
#         data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
#         image_root = '/wdata/roironen/Data/AirMSPINEW/'
#         mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#         image_size = [350, 350]
#     else:
#         NotImplementedError()
#     images_mapping_list = []
#     for _, map in mapping.items():
#         voxels_list = []
#         v = map.values()
#         voxels = np.array(list(v),dtype=object)
#         ctr = 0
#         for i, voxel in enumerate(voxels):
#             if len(voxel)>0:
#                 pixels = np.unravel_index(voxel, np.array(image_size))
#                 mean_px = np.mean(pixels,1)
#                 voxels_list.append(mean_px)
#             else:
#                 ctr +=1
#                 voxels_list.append([-100000,-100000])
#         images_mapping_list.append(voxels_list)
#
#     #pixel_centers = sio.loadmat('/wdata/roironen/Data/pixel_centers.mat')['xpc']
#     print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
#     image_train_paths = [f for f in glob.glob(os.path.join(image_root, "*.pkl"))]
#     cloud_train_path = data_root
#     train_len = cfg.data.n_training if cfg.data.n_training>0 else len(image_train_paths)
#
#     image_train_paths = image_train_paths[:train_len]
#     # cloud_train_paths = cloud_train_paths[:train_len]
#     n_cam = cfg.data.n_cam
#     mean = cfg.data.mean
#     std = cfg.data.std
#     # rand_cam = cfg.data.rand_cam
#     train_dataset = AirMSPIDataset(
#             cloud_train_path,
#         image_train_paths,
#         mapping=images_mapping_list,
#         n_cam=n_cam,
#         mask_type=cfg.ct_net.mask_type,
#         mean=mean,
#         std=std,
#     dataset_name = dataset_name,
#         drop_index = cfg.data.drop_index,
#         #pixel_centers=pixel_centers
#     )
#
#
#
#     return train_dataset, train_dataset,  n_cam
#
#
# class AirMSPIDataset(Dataset):
#     def __init__(self, cloud_dir,image_dir, n_cam,mapping, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
#         self.cloud_dir = cloud_dir
#         self.mapping = mapping
#         self.image_dir = image_dir
#         self.mask_type = mask_type
#         self.n_cam = n_cam
#         self.mean = mean
#         self.std = std
#         self.dataset_name = dataset_name
#         self.drop_index = drop_index
#         if self.n_cam != 9 and self.drop_index>-1:
#             self.mapping.pop(drop_index)
#             #self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)
#
#     def __len__(self):
#         return len(self.cloud_dir)
#
#     def __getitem__(self, idx):
#         image_dir = self.image_dir[idx]
#         image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
#         cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")
#
#
#         with open(cloud_path, 'rb') as f:
#             data = pickle.load(f)
#         with open(image_dir, 'rb') as f:
#             images = pickle.load(f)['images']
#         if self.n_cam!=9:
#             images = np.delete(images,self.drop_index,0)
#         mask = None
#         if self.mask_type == 'space_carving':
#             mask = data['mask']
#         elif self.mask_type == 'space_carving_morph':
#             mask = data['mask_morph']
#         images -= self.mean
#         images /= self.std
#         grid = data['grid']
#         # grid = data['net_grid']
#         # if hasattr(data, 'image_sizes'):
#         #     image_sizes = data['image_sizes']
#         # else:
#         #     image_sizes = [image.shape for image in images]
#         extinction = data['ext'] / 10
#
#         images_mapping_list = [ np.array(map)[mask.ravel()] for map in self.mapping]
#
#         return images, extinction, grid, images_mapping_list, mask
#
#
#
# def get_airmspi_datasetsV2(
#     cfg,
#     data_root: str = DEFAULT_DATA_ROOT,
# ) -> Tuple[Dataset, Dataset, int]:
#     """
#     Obtains the training and validation dataset object for a dataset specified
#     with the `dataset_name` argument.
#
#     Args:
#         dataset_name: The name of the dataset to load.
#         image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
#         data_root: The root folder at which the data is stored.
#
#     Returns:
#         train_dataset: The training dataset object.
#         val_dataset: The validation dataset object.
#         test_dataset: The testing dataset object.
#     """
#     dataset_name = cfg.data.dataset_name
#
#     if dataset_name not in ALL_DATASETS_AIRMSPI:
#         raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")
#
#     if dataset_name == 'BOMEX_9cams':
#         data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
#         image_root = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/AirMSPI/test/LOW_SC/AIRMSPI_IMAGES_LWC_LOW_SC'
#         mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#         image_size = [350, 350]
#     else:
#         NotImplementedError()
#     images_mapping_list = []
#     pixel_centers_list = []
#     pixel_centers = sio.loadmat('/wdata/roironen/Data/pixel_centers.mat')['xpc']
#     camera_ind = 0
#     for _, map in mapping.items():
#         voxels_list = []
#         pixel_list = []
#         v = map.values()
#         voxels = np.array(list(v),dtype=object)
#         for i, voxel in enumerate(voxels):
#             if len(voxel)>0:
#                 pixels = np.unravel_index(voxel, np.array(image_size))
#                 mean_px = np.mean(pixels,1)
#                 voxels_list.append(mean_px)
#                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
#             else:
#                 voxels_list.append([-100000,-100000])
#                 pixel_list.append([-10000, -10000, -10000])
#
#         camera_ind += 1
#         images_mapping_list.append(voxels_list)
#         pixel_centers_list.append(pixel_list)
#
#     print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
#     image_train_paths = [f for f in glob.glob(os.path.join(image_root, "*.pkl"))]
#     cloud_train_path = data_root
#     train_len = cfg.data.n_training if cfg.data.n_training>0 else len(image_train_paths)
#
#     image_train_paths = image_train_paths[:train_len]
#     # cloud_train_paths = cloud_train_paths[:train_len]
#     n_cam = cfg.data.n_cam
#     mean = cfg.data.mean
#     std = cfg.data.std
#     # rand_cam = cfg.data.rand_cam
#     train_dataset = AirMSPIDatasetV2(
#             cloud_train_path,
#         image_train_paths,
#         mapping=images_mapping_list,
#         n_cam=n_cam,
#         mask_type=cfg.ct_net.mask_type,
#         mean=mean,
#         std=std,
#     dataset_name = dataset_name,
#         drop_index = cfg.data.drop_index,
#         pixel_centers=pixel_centers_list
#     )
#
#
#
#     return train_dataset, train_dataset,  n_cam
#
#
# class AirMSPIDatasetV2(Dataset):
#     def __init__(self, cloud_dir,image_dir, n_cam,mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
#         self.cloud_dir = cloud_dir
#         self.mapping = mapping
#         self.image_dir = image_dir
#         self.mask_type = mask_type
#         self.n_cam = n_cam
#         self.mean = mean
#         self.std = std
#         self.dataset_name = dataset_name
#         self.pixel_centers = pixel_centers
#         self.drop_index = drop_index
#         if self.n_cam != 9 and self.drop_index>-1:
#             self.mapping.pop(drop_index)
#             self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)
#
#     def __len__(self):
#         return len(self.cloud_dir)
#
#     def __getitem__(self, idx):
#         image_dir = self.image_dir[idx]
#         image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
#         cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")
#
#
#         with open(cloud_path, 'rb') as f:
#             data = pickle.load(f)
#         with open(image_dir, 'rb') as f:
#             images = pickle.load(f)['images']
#         if self.n_cam!=9:
#             images = np.delete(images,self.drop_index,0)
#         mask = None
#         if self.mask_type == 'space_carving':
#             mask = data['mask']
#         elif self.mask_type == 'space_carving_morph':
#             mask = data['mask_morph']
#         images -= self.mean
#         images /= self.std
#         grid = data['grid']
#         # grid = data['net_grid']
#         # if hasattr(data, 'image_sizes'):
#         #     image_sizes = data['image_sizes']
#         # else:
#         #     image_sizes = [image.shape for image in images]
#         extinction = data['ext'] / 10
#
#         images_mapping_list = [ np.array(map)[mask.ravel()] for map in self.mapping]
#         pixel_centers = [ np.array(centers)[mask.ravel()] for centers in self.pixel_centers]
#
#
#         return images, extinction, grid, images_mapping_list, pixel_centers, mask
#

def get_airmspi_datasets(
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

    if dataset_name not in ALL_DATASETS_AIRMSPI:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'AirMSPI_BOMEX_aux_9cams':
        cloud_train_path = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/train') # use 3D clouds from here
        image_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256/AirMSPI_pushbroom_camera') # use push-broom rendered images
        mapping_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/voxel_pixel_list*.pkl'))]
        pixel_center_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/pixel_centers_*.mat'))]
        image_size = [350, 350]
        image_train_paths = [f for f in glob.glob(os.path.join(image_root, "SIMULATED_AIRMSPI_TRAIN*"))]
        image_train_paths = [glob.glob(os.path.join(f, "*.pkl")) for f in image_train_paths]
        cloud_adj = 10
        with open(os.path.join(data_root, 'AirMSPI/training/images_mapping.pkl'), 'rb') as f:
            images_mapping_lists = pickle.load(f)  # pre-computed voxel-pixel mapping
        with open(os.path.join(data_root, 'AirMSPI/training/pixel_centers.pkl'), 'rb') as f:
            pixel_centers_lists = pickle.load(f)  # pre-computed 3D pixel center
    elif dataset_name == 'AirMSPI_BOMEX_50CCN_9cams':
        cloud_train_path = os.path.join(data_root,
                                        'BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/train')  # use 3D clouds from here
        # cloud_train_path += os.path.join(data_root,
        #                                 'DYCOMS_RF02_50CCN_64x64x159_50m/10cameras_20m/train')

        image_root = os.path.join(data_root, 'BOMEX_128x128x100_50CCN_50m_micro_256/renderings_BOMEX_32x32x64_50CCN_50m')  # use push-broom rendered images
        # image_root += os.path.join('/wdata_visl/NEW_BOMEX/renderings_DYCOMS_RF02_64x159_50m_50CCN')  # use push-broom rendered images
        mapping_paths = [f for f in glob.glob(os.path.join(data_root, 'airmspi_projections/train_32x32x64/BOMEX_32x32x64_50CCN_50m_voxel_pixel_maps/voxel_pixel_list*.pkl'))]
        pixel_center_paths = [f for f in
                              glob.glob(os.path.join(data_root, 'AirMSPI/training/pixel_centers_*.mat'))]
        image_size = [350, 350]
        image_train_paths = [f for f in glob.glob(os.path.join(image_root, "SIMULATED_AIRMSPI_TRAIN*"))]
        image_train_paths = [glob.glob(os.path.join(f, "*.pkl")) for f in image_train_paths]
        cloud_adj = 1

        with open(os.path.join(data_root, 'AirMSPI/training/32x32x64_images_mapping.pkl'), 'rb') as f:
            images_mapping_lists = pickle.load(f)  # pre-computed voxel-pixel mapping
        with open(os.path.join(data_root, 'AirMSPI/training/32x32x64_pixel_centers.pkl'), 'rb') as f:
            pixel_centers_lists = pickle.load(f)  # pre-computed 3D pixel center

    else:
        NotImplementedError()
    ## building map if necessary
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # with open(os.path.join(data_root, 'AirMSPI/training/32x32x64_images_mapping.pkl'), 'wb') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(data_root, 'AirMSPI/training/32x32x64_pixel_centers.pkl'), 'wb') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)


    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")



    assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    train_dataset = AirMSPIDataset(
            cloud_train_path,
        image_train_paths,
        mapping=images_mapping_lists,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
        drop_index = cfg.data.drop_index,
        pixel_centers=pixel_centers_lists,
        cloud_adj = cloud_adj
    )

    return train_dataset, train_dataset


def get_real_world_airmspi_datasets_ft(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset]:
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

    if dataset_name not in ALL_DATASETS_AIRMSPI:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == '32N123W_experiment_all_clouds':
        image_paths = [os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/airmspi_9images.mat"),
                       os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/airmspi_9images.mat"),
                       os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/airmspi_9images.mat"),
                       os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/airmspi_9images.mat")
                       ]

        mapping_paths = [os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/images_mapping.pkl"),
                         os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/images_mapping.pkl"),
                         os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/images_mapping.pkl"),
                         os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/images_mapping.pkl")
                         ]

        pixel_centers_paths = [os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/pixel_centers.pkl"),
                               os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/pixel_centers.pkl"),
                               os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/pixel_centers.pkl"),
                               os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/pixel_centers.pkl")
                               ]

        mask_paths = [os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/mask_72x72x32_vox50x50x40m.mat"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/mask_60x60x32_vox50x50x40m.mat"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/mask_72x72x32_vox50x50x40m.mat"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/mask_72x72x32_vox50x50x40m.mat")
                      ]

        shdom_proj_paths = [os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/projections"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/projections"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/projections"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/projections")
                      ]

        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = [72,60,72,72]
        ny = [72,60,72,72]
        nz = 32

    elif dataset_name == '32N123W_experiment_234_clouds':
        image_paths = [
                       os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/airmspi_9images.mat"),
                       os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/airmspi_9images.mat"),
                       os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/airmspi_9images.mat")
                       ]

        mapping_paths = [
                         os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/images_mapping.pkl"),
                         os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/images_mapping.pkl"),
                         os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/images_mapping.pkl")
                         ]

        pixel_centers_paths = [
                               os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/pixel_centers.pkl"),
                               os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/pixel_centers.pkl"),
                               os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/pixel_centers.pkl")
                               ]

        mask_paths = [
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/mask_60x60x32_vox50x50x40m.mat"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/mask_72x72x32_vox50x50x40m.mat"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/mask_72x72x32_vox50x50x40m.mat")
                      ]

        shdom_proj_paths = [
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/projections"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud3/projections"),
                      os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud4/projections")
                      ]

        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = [60,72,72]
        ny = [60,72,72]
        nz = 32

    else:
        NotImplementedError()
    ## building map if necessary

    # mapping_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/32N123W_experiment_cloud4/voxel_pixel_list*.pkl'))]
    # pixel_center_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/32N123W_experiment_cloud4/pixel_centers.mat'))]
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # image_size = [450,450]
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    # with open(os.path.join(data_root, 'AirMSPI/test/32N123W_experiment_cloud4/72x72x32_images_mapping.pkl'), 'wb') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(data_root, 'AirMSPI/test/32N123W_experiment_cloud4/72x72x32_pixel_centers.pkl'), 'wb') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)
    train_len = cfg.data.n_training if cfg.data.n_training > 0 else len(mapping_paths)
    mapping_paths = mapping_paths[:train_len]
    pixel_centers_paths = pixel_centers_paths[:train_len]
    mask_paths = mask_paths[:train_len]
    shdom_proj_paths = shdom_proj_paths[:train_len]
    image_paths = image_paths[:train_len]
    nx = nx[:train_len]
    ny = ny[:train_len]

    images_mapping_lists = []
    pixel_centers_lists = []
    shdom_proj_lists = []
    mask_lists = []

    for mapping_path, pixel_centers_path, mask_path, shdom_proj_path in zip(mapping_paths, pixel_centers_paths, mask_paths, shdom_proj_paths):
        mask = sio.loadmat(mask_path)['mask']>0
        mask_lists.append(mask)
        with open(mapping_path, 'rb') as f:
            map = pickle.load(f)
            # print(np.array(map).shape)
            map = np.array(map).squeeze()#[:,mask.ravel(),:]
            if cfg.data.n_cam != 9 and cfg.data.drop_index > -1:
                map = np.delete(map,cfg.data.drop_index,0)
            # print(np.array(map).shape)
            images_mapping_lists.append(map)
        with open(pixel_centers_path, 'rb') as f:
            centers = pickle.load(f)
            # print(np.array(centers).shape)
            centers = np.array(centers).squeeze()#[:,mask.ravel(),:]
            if cfg.data.n_cam != 9 and cfg.data.drop_index > -1:
                centers = np.delete(centers,cfg.data.drop_index,0)
            # print(np.array(centers).shape)
            pixel_centers_lists.append(centers)
        with open(shdom_proj_path, 'rb') as pickle_file:
            projection_list = pickle.load(pickle_file)['projections']
            if cfg.data.n_cam != 9 and cfg.data.drop_index > -1:
                projection_list.pop(cfg.data.drop_index)
            shdom_proj_lists.append(projection_list)

    # images_mapping_list = [[np.array(map) for map in images_mapping_list]]
    # pixel_centers_list = [[np.array(centers) for centers in pixel_centers_list]]
    grids = []
    for nx_i, ny_i in zip(nx, ny):
        gx = np.linspace(0, dx * (nx_i-1), nx_i, dtype=np.float32)
        gy = np.linspace(0, dy * (ny_i-1), ny_i, dtype=np.float32)
        gz = np.linspace(0, dz * (nz-1), nz, dtype=np.float32)
        grids.append([np.array([gx, gy, gz])])

    # masks = []
    # for mask_path in mask_paths:
    #     masks.append(sio.loadmat(mask_path)['mask'])

    # assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    dataset = AirMSPIDataset_ft(
        image_dir = image_paths,
        mapping=images_mapping_lists,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
        drop_index = cfg.data.drop_index,
        pixel_centers=pixel_centers_lists,
        shdom_proj_lists = shdom_proj_lists,
        mask = mask_lists,
        grid = grids

    )

    return dataset


def get_real_world_airmspi_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset]:
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

    if dataset_name not in ALL_DATASETS_AIRMSPI:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")
    projection_list = None
    if dataset_name == '32N123W_experiment_cloud1':
        image_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/airmspi_9images.mat")
        mapping_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/images_mapping.pkl")
        pixel_centers_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/pixel_centers.pkl")
        mask_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/mask_72x72x32_vox50x50x40m.mat")
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 72
        ny = 72
        nz = 32

        shdom_proj_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud1/projections")
        with open(shdom_proj_path, 'rb') as pickle_file:
            projection_list = pickle.load(pickle_file)['projections']
            if cfg.data.n_cam != 9 and cfg.data.drop_index > -1 and cfg.rerender:
                projection_list = [projection_list[cfg.data.drop_index]]

    elif dataset_name == '32N123W_experiment_cloud2':
        image_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/airmspi_9images.mat")
        mapping_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/images_mapping.pkl")
        pixel_centers_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/pixel_centers.pkl")
        mask_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/mask_60x60x32_vox50x50x40m.mat")
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 60
        ny = 60
        nz = 32
    elif dataset_name == '18S8E_experiment':
        image_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/airmspi_9images.mat")
        mapping_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/images_mapping.pkl")
        pixel_centers_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/pixel_centers.pkl")
        mask_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/mask_52x52x32_vox50x50x40m.mat")
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 52
        ny = 52
        nz = 32

    else:
        NotImplementedError()
    ## building map if necessary
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_images_mapping_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_pixel_centers_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)
    with open(mapping_path, 'rb') as f:
        images_mapping_list = pickle.load(f)
    with open(pixel_centers_path, 'rb') as f:
        pixel_centers_list = pickle.load(f)

    # images_mapping_list = [[np.array(map) for map in images_mapping_list]]
    # pixel_centers_list = [[np.array(centers) for centers in pixel_centers_list]]

    gx = np.linspace(0, dx * (nx-1), nx, dtype=np.float32)
    gy = np.linspace(0, dy * (ny-1), ny, dtype=np.float32)
    gz = np.linspace(0, dz * (nz-1), nz, dtype=np.float32)
    grid = [np.array([gx, gy, gz])]
    mask = sio.loadmat(mask_path)['mask']

    # assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    dataset = AirMSPIDataset_test(
        image_dir = image_path,
        mapping=images_mapping_list,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
        drop_index = cfg.data.drop_index,
        pixel_centers=pixel_centers_list,
        mask = mask,
        grid = grid,
        projection_list=projection_list

    )

    return dataset

class AirMSPIDataset(Dataset):
    def __init__(self, cloud_dir,image_dir, n_cam,mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1, cloud_adj=1):
        self.cloud_dir = cloud_dir
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        self.cloud_adj = cloud_adj
        if self.n_cam != 9 and self.drop_index>-1:
            for map in self.mapping:
                map.pop(drop_index)
            self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,1)

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        geometry_ind = np.random.randint(len(self.image_dir))
        image_dir = self.image_dir[geometry_ind][idx]
        image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
        cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")

        try:
            with open(cloud_path, 'rb') as f:
                data = pickle.load(f)
            with open(image_dir, 'rb') as f:
                images = pickle.load(f)['images']
        except:
            return None, None, None, None, None, None
        if self.n_cam != 9:
            images = np.delete(images, self.drop_index,0)
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        images -= self.mean
        images /= self.std
        grid = data['grid']
        extinction = data['ext'] / self.cloud_adj # convert BOMEX clouds to BOMEX_aux clouds

        images_mapping_list = [np.array(map)[mask.ravel()] for map in self.mapping[geometry_ind]]
        pixel_centers = [np.array(centers)[mask.ravel()] for centers in self.pixel_centers[geometry_ind]]

        return images, extinction, grid, images_mapping_list, pixel_centers, mask

class AirMSPIDataset_ft(Dataset):
    def __init__(self, image_dir, n_cam, mapping, pixel_centers, shdom_proj_lists, mask_type=None, mean=0, std=1, dataset_name='',
                 drop_index=-1, mask = None, grid = None):
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        self.mask = mask
        self.grid =grid
        self.shdom_proj_lists = shdom_proj_lists

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        images = sio.loadmat(self.image_dir[idx])['croped_airmspi_images']
        if self.n_cam != 9:
            images = np.delete(images, self.drop_index, 0)

        if self.mask_type == 'space_carving':
            mask = self.mask[idx]
        else:
            mask = np.ones_like(self.grid)>0
        images -= self.mean
        images /= self.std
        # mapping = self.mapping[idx]
        # pixel_centers = self.pixel_centers[idx]
        # images_mapping_list = [[np.array(map) for map in mapping]]
        # pixel_centers_list = [[np.array(centers) for centers in pixel_centers]]
        images_mapping_list =  self.mapping[idx]# [np.array(map)[mask.ravel()] for map in]
        pixel_centers_list = self.pixel_centers[idx] #[np.array(centers)[mask.ravel()] for centers in ]
        shdom_proj_list = self.shdom_proj_lists[idx]
        grid = self.grid[idx][0]
        return images, grid, images_mapping_list, pixel_centers_list, mask, shdom_proj_list


class AirMSPIDataset_test(Dataset):
    def __init__(self, image_dir, n_cam, mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='',
                 drop_index=-1, mask = None, grid = None,projection_list=None):
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        self.mask = mask
        self.grid =grid
        self.projection_list = projection_list
        if self.n_cam != 9 and self.drop_index>-1:
            self.mapping.pop(self.drop_index)
            self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)


    def __getitem__(self, idx):
        gt_image=None
        images = sio.loadmat(self.image_dir)['croped_airmspi_images']
        if self.n_cam != 9:
            if self.projection_list is not None:
                gt_image = images[self.drop_index][None]
                gt_image -= self.mean
                gt_image /= self.std
            images = np.delete(images, self.drop_index, 0)
        mask = None
        if self.mask_type == 'space_carving':
            mask = self.mask
        else:
            mask = np.ones_like(self.grid)
        images -= self.mean
        images /= self.std
        images_mapping_list = [[np.array(map) for map in self.mapping]]
        pixel_centers_list = [[np.array(centers) for centers in self.pixel_centers]]

        return images, self.grid, images_mapping_list, pixel_centers_list, mask, gt_image,self.projection_list
