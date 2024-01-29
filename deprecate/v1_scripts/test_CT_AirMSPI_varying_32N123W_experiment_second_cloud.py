# This file contains the main script for VIP-CT evaluation on AirMSPI data.
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

import collections
import os, time
import pickle
import warnings
# import sys
# sys.path.insert(0, '/home/roironen/pytorch3d/projects/')
import hydra
import numpy as np
import torch
from ProbCT.util.visualization import SummaryWriter
from dataset import get_cloud_datasets, trivial_collate
from CTnet import *
from util.stats import Stats
from omegaconf import OmegaConf
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from cameras import AirMSPICameras
import scipy.io as sio
from ProbCT.scene.volumes import Volumes

relative_error = lambda ext_est, ext_gt, eps=1e-6 : torch.norm(ext_est.view(-1) - ext_gt.view(-1),p=1) / (torch.norm(ext_gt.view(-1),p=1) + eps)
mass_error = lambda ext_est, ext_gt, eps=1e-6 : (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

def show_scatter_plot(gt_param, est_param):
    gt_param = gt_param.detach().cpu().numpy().ravel()
    est_param = est_param.detach().cpu().numpy().ravel()
    max_val = max(gt_param.max(), est_param.max())
    fig, ax = plt.subplots()
    ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
    ax.set_xlim([0, 1.1 * max_val])
    ax.set_ylim([0, 1.1 * max_val])
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')
    plt.show()

def show_scatter_plot_altitute(gt_param, est_param):
    import matplotlib.cm as cm
    gt_param = gt_param.detach().cpu().numpy()
    est_param = est_param.detach().cpu().numpy()
    colors = cm.rainbow(np.linspace(0, 1, gt_param.shape[-1]))

    max_val = max(gt_param.max(), est_param.max())
    fig, ax = plt.subplots()
    for i, c in enumerate(colors):
        if i>10:
            ax.scatter(gt_param[...,i].ravel(), est_param[...,i].ravel(),
                   facecolors='none', edgecolors=c, label=i)
        else:
            ax.scatter(gt_param[..., i].ravel(), est_param[..., i].ravel(),
                       facecolors='none', edgecolors=c)
    ax.set_xlim([0, 1.1 * max_val])
    ax.set_ylim([0, 1.1 * max_val])
    ax.legend(loc='best',fontsize='small')
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')
    plt.show()

def volume_plot(gt_param, est_param):
    gt_param = gt_param.detach().cpu().numpy()
    est_param = est_param.detach().cpu().numpy()
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("GT")
    ax.voxels(gt_param)
    plt.show()
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("Est.")
    ax.voxels(est_param)
    plt.show()

@hydra.main(config_path=CONFIG_DIR, config_name="AirMSPI_test_varying")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available() and cfg.debug == False:
        n_device = torch.cuda.device_count()
        cfg.gpu = 0 if n_device==1 else cfg.gpu
        device = f"cuda:{cfg.gpu}"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"
        # Init the visualization visdom env.
    log_dir = os.getcwd()
    log_dir = log_dir.replace('outputs','test_results')
    writer = None #SummaryWriter(log_dir)
    results_dir = log_dir #os.path.join(log_dir, 'test_results')
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(results_dir) > 0:
        # Make the root of the experiment directory.
        # checkpoint_dir = os.path.split(checkpoint_path)
        os.makedirs(results_dir, exist_ok=True)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg=OmegaConf.merge(net_cfg,cfg)
    # DATA_DIR = os.path.join(current_dir, "data")
    # _, val_dataset, n_cam = get_cloud_datasets(
    #     cfg=cfg
    # )

    # Initialize the Radiance Field model.
    model = CTnetAirMSPI(cfg=cfg, n_cam=cfg.data.n_cam)

    # Move the model to the relevant device.

    # Resume training if requested.
    assert os.path.isfile(checkpoint_resume_path)
    print(f"Resuming from checkpoint {checkpoint_resume_path}.")
    loaded_data = torch.load(checkpoint_resume_path, map_location=device)
    model.load_state_dict(loaded_data["model"])
    model.to(device)
    # stats = pickle.loads(loaded_data["stats"])
    # print(f"   => resuming from epoch {stats.epoch}.")
    # optimizer_state_dict = loaded_data["optimizer"]
    # start_epoch = stats.epoch
    # Init stats to None before loading.
    # stats = None
    # optimizer_state_dict = None
    # start_epoch = 0



    # Initialize the optimizer.
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=cfg.optimizer.lr,
    # )

    # Load the optimizer state dict in case we are resuming.
    # if optimizer_state_dict is not None:
    #     optimizer.load_state_dict(optimizer_state_dict)
    #     optimizer.last_epoch = start_epoch

    # Init the stats object.
    stats = Stats(["loss", "relative_error", "lr", "max_memory", "sec/it"])


    # The validation dataloader is just an endless stream of random samples.

    # err = torch.nn.MSELoss()
    # err = torch.nn.L1Loss(reduction='sum')
    # Set the model to the training mode.
    model.eval().float()
    model.eval()

    # Run the main training loop.
    iteration = -1


    # Validation
    # loss_val = 0
    relative_err= []
    relative_mass_err = []
    batch_time_net = []
    val_image = sio.loadmat('/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/32N123W_experiment/croped_airmspi_9images_for_Roi.mat')['croped_airmspi_images'] # 32N123W_experiment

    if cfg.data.n_cam != 9:
        val_image = np.delete(val_image, cfg.data.drop_index, 0)
    val_image = torch.tensor(val_image,device=device).float()[None]

    masks = sio.loadmat('/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/32N123W_experiment/mask_60x60x32_vox50x50x40m.mat')['mask'] # 32N123W_experiment


    # mapping_path = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/32N123W_experiment/voxel_pixel_list60x60x32_BOMEX_img350x350.pkl' # 32N123W_experiment
    # with open(mapping_path, 'rb') as f:
    #     mapping = pickle.load(f)
    # # images_mapping_list = sio.loadmat(mapping_path)['map']
    # pixel_center_path = '/wdata/roironen/Data/32N132W_second_cloud_shdom_measurements_pixels_center.mat'
    # pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    # image_size = [350, 350]
    # images_mapping_list = []
    # pixel_centers_list = []
    # camera_ind = 0
    # for _, map in mapping.items():
    #     voxels_list = []
    #     pixel_list = []
    #     v = map.values()
    #     voxels = np.array(list(v),dtype=object)
    #     for i, voxel in enumerate(voxels):
    #         if len(voxel)>0:
    #             pixels = np.unravel_index(voxel, np.array(image_size))
    #             mean_px = np.mean(pixels,1)
    #             voxels_list.append(mean_px)
    #             pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #         else:
    #             voxels_list.append([-100000,-100000])
    #             pixel_list.append([-10000, -10000, -10000])
    #
    #     camera_ind += 1
    #     images_mapping_list.append(voxels_list)
    #     pixel_centers_list.append(pixel_list)
    #
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/32N123W_rebat_images_mapping_lists60x60x32_18S8E_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_list, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/32N123W_rebat_pixel_centers_lists60x60x32_18S8E_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_list, f, pickle.HIGHEST_PROTOCOL)

    with open('/wdata/roironen/Data/AirMSPI-Varying/test/32N123W_rebat_images_mapping_lists60x60x32_18S8E_img350x350.pkl', 'rb') as f:
        images_mapping_list = pickle.load(f)
    with open('/wdata/roironen/Data/AirMSPI-Varying/test/32N123W_rebat_pixel_centers_lists60x60x32_18S8E_img350x350.pkl', 'rb') as f:
        pixel_centers_list = pickle.load(f)

    if cfg.data.n_cam != 9:
        # print(len(images_mapping_list))
        images_mapping_list.pop(cfg.data.drop_index)
        # print(len(images_mapping_list))
        pixel_centers_list = np.delete(pixel_centers_list, cfg.data.drop_index, 0)


    mean = cfg.data.mean
    std = cfg.data.std
    val_image -= mean
    val_image /= std
    images_mapping_list = [[np.array(map) for map in images_mapping_list]]
    pixel_centers_list = [[np.array(centers) for centers in pixel_centers_list]]

    masks = torch.tensor(masks,device=device)[None]
    # gx = np.linspace(-20*0.05,0.05 * 52,72, dtype=np.float32)
    # gy = np.linspace(-20*0.05, 0.05 * 52, 72, dtype=np.float32)
    gx = np.linspace(0,0.05 * 60,60, dtype=np.float32)
    gy = np.linspace(0, 0.05 * 60, 60, dtype=np.float32)
    gz = np.linspace(0, 0.04 * 32, 32, dtype=np.float32)
    grid = [np.array([gx,gy,gz])]
    val_volume = Volumes(torch.unsqueeze(torch.tensor(masks, device=device).float(), 1), grid)
    val_camera = AirMSPICameras(mapping=torch.tensor(images_mapping_list, device=device).float(),
                                  centers=torch.tensor(pixel_centers_list).float(),
                             device=device)


# Activate eval mode of the model (lets us do a full rendering pass).
    with torch.no_grad():
        est_vols = torch.zeros(masks.shape, device=masks.device)
        # n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
        # if n_points_mask > cfg.min_mask_points:
        net_start_time = time.time()

        val_out = model(
            val_camera,
            val_image,
            val_volume,
            masks
        )
        for i, (out_vol, m) in enumerate(zip(val_out["output"],masks)):
            if m is None:
                est_vols[i] = out_vol.squeeze(1)
            else:
                # m = m.reshape(-1)
                est_vols[i][m] = out_vol.squeeze(1)

        time_net = time.time() - net_start_time
        assert len(est_vols)==1 ##TODO support validation with batch larger than 1
        est_vols[est_vols<0] = 0

        airmspi_cloud = {'cloud':est_vols[0].cpu().numpy()}

        sio.savemat('airmspi_try1.mat', airmspi_cloud)
        # est_vols[gt_vol==0] = 0
        # loss_val += err(est_vols, gt_vol)
        # loss_val += l1(val_out["output"], val_out["volume"]) / torch.sum(val_out["volume"]+1000)
        # print(f'{relative_error(ext_est=est_vols,ext_gt=gt_vol)}, {n_points_mask}')
        # if relative_error(ext_est=est_vols,ext_gt=gt_vol)>2:
        #     print()

        # relative_err.append(relative_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
        # relative_mass_err.append(mass_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
        batch_time_net.append(time_net)
        # if False:
        #     show_scatter_plot(gt_vol,est_vols)
        #     show_scatter_plot_altitute(gt_vol,est_vols)
        #     volume_plot(gt_vol,est_vols)
        # if writer:
        #     writer._iter = iteration
        #     writer._dataset = 'val'  # .format(val_i)
        #     if val_i in val_scatter_ind:
        #         writer.monitor_scatter_plot(est_vols, gt_vol,ind=val_i)
        # Update stats with the validation metrics.


    # loss_val /= (val_i + 1)
    # relative_err = np.array(relative_err)
    # relative_mass_err =np.array(relative_mass_err)
    batch_time_net = np.array(batch_time_net)
    # print(f'mean relative error {np.mean(relative_err)} with std of {np.std(relative_err)} for {(val_i + 1)} clouds')
    # masked = relative_err<2
    # relative_err1 = relative_err[masked]
    # print(f'mean relative error w/o outliers {np.mean(relative_err1)} with std of {np.std(relative_err1)} for {relative_err1.shape[0]} clouds')
    #
    # print(f'mean relative mass error {np.mean(relative_mass_err)} with std of {np.std(relative_mass_err)} for {(val_i + 1)} clouds')
    # relative_mass_err1 = relative_mass_err[masked]
    # print(f'mean relative mass error w/o outliers {np.mean(relative_mass_err1)} with std of {np.std(relative_mass_err1)} for {relative_mass_err1.shape[0]} clouds')

    print(f'Mean time = {np.mean(batch_time_net)}')


if __name__ == "__main__":
    main()


