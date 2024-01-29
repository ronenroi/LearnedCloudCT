# This file contains a work in progress code.
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
from LearnedCloudCT.ProbCT.util.visualization import SummaryWriter
from LearnedCloudCT.dataloader.microphysics_dataset import get_cloud_microphysics_datasets, trivial_collate
from LearnedCloudCT.ProbCT.CTnet import *
from LearnedCloudCT.ProbCT.util.stats import Stats
from omegaconf import OmegaConf
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from LearnedCloudCT.scene.volumes import Volumes
from LearnedCloudCT.scene.cameras import PerspectiveCameras

relative_error = lambda ext_est, ext_gt, eps=1e-6 : torch.norm(ext_est.view(-1) - ext_gt.view(-1),p=1) / (torch.norm(ext_gt.view(-1),p=1) + eps)
mass_error = lambda ext_est, ext_gt, eps=1e-6 : (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../configs")

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

@hydra.main(config_path=CONFIG_DIR, config_name="basic_test")
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
    _, val_dataset = get_cloud_microphysics_datasets(
        cfg=cfg
    )

    # Initialize the Radiance Field model.
    model = CTnetMicrophysics(cfg=cfg, n_cam=cfg.data.n_cam)

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
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
        # sampler=torch.utils.data.RandomSampler(
        #     val_dataset,
        #     replacement=True,
        #     num_samples=cfg.optimizer.max_epochs,
        # ),
    )
    # err = torch.nn.MSELoss()
    # err = torch.nn.L1Loss(reduction='sum')
    # Set the model to the training mode.
    model.eval().float()
    model.eval()

    # Run the main training loop.
    iteration = -1
    if writer:
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]

    # Validation
    # loss_val = 0
    lwc_relative_err= []
    reff_relative_err= []
    veff_relative_err= []
    relative_mass_err = []
    batch_time_net = []
    val_i = 0
    for val_i, val_batch in enumerate(val_dataloader):

    # val_batch = next(val_dataloader.__iter__())

        val_image, microphysics, grid, image_sizes, projection_matrix, camera_center, masks = val_batch  # [0]#.values()
        val_image = torch.tensor(val_image, device=device).float()
        val_volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)

        val_camera = PerspectiveCameras(image_size=image_sizes, P=torch.tensor(projection_matrix, device=device).float(),
                                        camera_center=torch.tensor(camera_center, device=device).float(), device=device)
        if model.val_mask_type == 'gt_mask':
            masks = val_volume.extinctions > 0 #val_volume._ext_thr
        else:
            masks = [torch.tensor(mask) if mask is not None else torch.ones(*extinction[0].shape,device=device, dtype=bool) for mask in masks]

    # Activate eval mode of the model (lets us do a full rendering pass).
        with torch.no_grad():
            est_vols = torch.zeros(val_volume.extinctions.numel(), device=val_volume.device).reshape(
                val_volume.extinctions.shape[0],val_volume.extinctions.shape[1], -1)
            n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
            if n_points_mask > cfg.min_mask_points:
                net_start_time = time.time()

                val_out = model(
                    val_camera,
                    val_image,
                    val_volume,
                    masks
                )
                if val_out['query_indices'] is None:
                    for i, (out_vol, m) in enumerate(zip(val_out["output"],masks)):
                        if m is None:
                            est_vols[i] = out_vol.squeeze(1)
                        else:
                            m = m.view(-1)
                            est_vols[i][m] = out_vol.squeeze(1)
                else:
                    for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                        est_vol[:,m]=out_vol.squeeze(1).T#.reshape(m.shape)[m]
                time_net = time.time() - net_start_time
            else:
                time_net = 0
            assert len(est_vols)==1 ##TODO support validation with batch larger than 1


            gt_vol = val_volume.extinctions[0].squeeze()
            est_vols = est_vols.squeeze().reshape(gt_vol.shape)
            # est_vols[gt_vol==0] = 0
            est_vols[est_vols<0] = 0

            est_lwc = est_vols[0]
            est_reff = est_vols[1]
            est_veff = est_vols[2]

            gt_lwc = gt_vol[0]
            gt_reff =  gt_vol[1]
            gt_veff = gt_vol[2]

            # loss_val += err(est_vols, gt_vol)
            # loss_val += l1(val_out["output"], val_out["volume"]) / torch.sum(val_out["volume"]+1000)
            print(f'LWC: {relative_error(ext_est=est_lwc,ext_gt=gt_lwc)}, {n_points_mask}')
            print(f'Reff: {relative_error(ext_est=est_reff,ext_gt=gt_reff)}, {n_points_mask}')
            print(f'Veff: {relative_error(ext_est=est_veff,ext_gt=gt_veff)}, {n_points_mask}')
            # if relative_error(ext_est=est_vols,ext_gt=gt_vol)>2:
            #     print()

            lwc_relative_err.append(relative_error(ext_est=est_lwc,ext_gt=gt_lwc).detach().cpu().numpy())#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            reff_relative_err.append(relative_error(ext_est=est_reff,ext_gt=gt_reff).detach().cpu().numpy())#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            veff_relative_err.append(relative_error(ext_est=est_veff,ext_gt=gt_veff).detach().cpu().numpy())#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            relative_mass_err.append(mass_error(ext_est=est_lwc,ext_gt=gt_lwc).detach().cpu().numpy())#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            batch_time_net.append(time_net)
            if False:
                show_scatter_plot(gt_vol,est_vols)
                show_scatter_plot_altitute(gt_vol,est_vols)
                volume_plot(gt_vol,est_vols)
            if writer:
                writer._iter = iteration
                writer._dataset = 'val'  # .format(val_i)
                if val_i in val_scatter_ind:
                    writer.monitor_scatter_plot(est_vols, gt_vol,ind=val_i)
            # Update stats with the validation metrics.


    # loss_val /= (val_i + 1)
    lwc_relative_err = np.array(lwc_relative_err)
    reff_relative_err = np.array(reff_relative_err)
    veff_relative_err = np.array(veff_relative_err)
    relative_mass_err =np.array(relative_mass_err)
    batch_time_net = np.array(batch_time_net)
    print(f'LWC: mean relative error {np.mean(lwc_relative_err)} with std of {np.std(lwc_relative_err)} for {(val_i + 1)} clouds')
    print(f'Reff: mean relative error {np.mean(reff_relative_err)} with std of {np.std(reff_relative_err)} for {(val_i + 1)} clouds')
    print(f'Veff: mean relative error {np.mean(veff_relative_err)} with std of {np.std(veff_relative_err)} for {(val_i + 1)} clouds')
    # masked = veff_relative_err<2
    # relative_err1 = relative_err[masked]
    # print(f'mean relative error w/o outliers {np.mean(relative_err1)} with std of {np.std(relative_err1)} for {relative_err1.shape[0]} clouds')
    #
    print(f'mean relative mass error {np.mean(relative_mass_err)} with std of {np.std(relative_mass_err)} for {(val_i + 1)} clouds')
    # relative_mass_err1 = relative_mass_err[masked]
    # print(f'mean relative mass error w/o outliers {np.mean(relative_mass_err1)} with std of {np.std(relative_mass_err1)} for {relative_mass_err1.shape[0]} clouds')

    print(f'Mean time = {np.mean(batch_time_net)} +- {np.std(batch_time_net)}')
    if writer:
        writer._iter = iteration
        writer._dataset = 'val'#.format(val_i)
        writer.monitor_loss(loss_val)
        # writer.monitor_scatterer_error(relative_mass_err, relative_err)
        # writer.monitor_images(val_image)


if __name__ == "__main__":
    main()


