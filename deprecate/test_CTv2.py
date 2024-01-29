# This file contains the main script for VIP-CT and ProbCT models.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite our papers described in the readme file.
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
import hydra
import numpy as np
import torch
from dataloader.dataset import get_cloud_datasets, trivial_collate
from LearnedCloudCT.ProbCT.CTnet import CTnet
from LearnedCloudCT.ProbCT.CTnetV2 import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
from LearnedCloudCT.ProbCT.util.plot_util import *
from LearnedCloudCT.ProbCT.util.visualization import SummaryWriter
import scipy.io as sio
from metrics.test_errors import *
from metrics.losses import *
from ProbCT import *
from LearnedCloudCT.scene.volumes import Volumes
from LearnedCloudCT.scene.cameras import PerspectiveCameras

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="vipctV2_test")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run
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

    log_dir = os.getcwd()
    log_dir = log_dir.replace('outputs','test_results')

    results_dir = log_dir
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(results_dir) > 0:
        # Make the root of the experiment directory
        os.makedirs(results_dir, exist_ok=True)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg = OmegaConf.merge(net_cfg,cfg)
    if cfg.show:
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    _, val_dataset = get_cloud_datasets(
        cfg=cfg
    )
    # Initialize VIP-CT model
    if cfg.version == 'V1':
        model = CTnet(cfg=cfg, n_cam=cfg.data.n_cam)
    else:
        model = CTnetV2(cfg=cfg, n_cam=cfg.data.n_cam)
    # Load model
    assert os.path.isfile(checkpoint_resume_path)
    print(f"Resuming from checkpoint {checkpoint_resume_path}.")
    loaded_data = torch.load(checkpoint_resume_path, map_location=device)
    model.load_state_dict(loaded_data["model"])
    model.to(device)

    # The validation dataloader is just an endless stream of random samples.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
    )

    # Set the model to eval mode.
    if cfg.mode == 'eval':
        model.eval().float()
    else:
        model.float()

    iteration = -1
    if writer:
        # show scatter plot of 5 clouds
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]

    relative_err = []
    l2_err = []
    relative_mass_err = []
    abs_err = []
    relative_voxel_err = []
    batch_time_net = []
    confidence_list = []
    est_list = []
    gt_list = []
    mask_list = []
    val_i = 0

    # Run the main training loop.
    for val_i, val_batch in enumerate(val_dataloader):
        val_image, extinction, grid, image_sizes, projection_matrix, camera_center, masks, _ = val_batch
        val_image = torch.tensor(np.array(val_image), device=device).float()
        val_volume = Volumes(torch.unsqueeze(torch.tensor(np.array(extinction), device=device).float(), 1), grid)
        val_camera = PerspectiveCameras(image_size=image_sizes, P=torch.tensor(projection_matrix, device=device).float(),
                                        camera_center=torch.tensor(camera_center, device=device).float(), device=device)
        if model.val_mask_type == 'gt_mask':
            masks = val_volume.extinctions > 0 #val_volume._ext_thr
        else:
            masks = [torch.tensor(mask) if mask is not None else torch.ones(*extinction[0].shape,device=device, dtype=bool) for mask in masks]

        with torch.no_grad():
            est_vols = torch.zeros(val_volume.extinctions.numel(), device=val_volume.device).reshape(
                val_volume.extinctions.shape[0], -1)
            n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
            conf_vol = torch.ones_like(est_vols[0]) * torch.nan

            # don't make inference on empty/small clouds
            if n_points_mask > cfg.min_mask_points:
                net_start_time = time.time()

                val_out = model(
                    val_camera,
                    val_image,
                    val_volume,
                    masks
                )
                if val_out["output"][0].shape[-1]>1:

                    val_out["output"], val_out["output_conf"], probs = get_pred_and_conf_from_discrete(val_out["output"],
                                                                                                cfg.cross_entropy.min,
                                                                                                cfg.cross_entropy.max,
                                                                                                cfg.cross_entropy.bins,
                                                                                                pred_type=cfg.ct_net.pred_type,
                                                                                                conf_type=cfg.ct_net.conf_type)
                else:
                    val_out["output_conf"] = None
                    probs = None
                if val_out['query_indices'] is None:
                    for i, (out_vol, m) in enumerate(zip(val_out["output"],masks)):
                        if m is None:
                            if len(out_vol.shape) == 1:
                                est_vols[i] = out_vol.reshape(-1)
                            else:  # value, std
                                est_vol[i] = out_vol[:, 0]
                                conf_vol = torch.ones_like(est_vol) * torch.nan
                                conf_vol[i] = out_vol[:, 1]
                        else:
                            m = m.view(-1)
                            if len(out_vol.shape) == 1:
                                est_vols[i][m] = out_vol.reshape(-1)
                            else:  # value, std
                                est_vol[i][m] = out_vol[:, 0]
                                conf_vol = torch.ones_like(est_vol) * torch.nan
                                conf_vol[i][m] = out_vol[:, 1]
                else:
                    for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                        if m.shape[-1] == 2:  # sequential querying
                            est_vol = est_vol.reshape(val_volume.extinctions.shape[2:])
                            conf_vol = torch.ones_like(est_vol) * torch.nan
                            for col_i in range(m.shape[0]):
                                est_vol[m[col_i, 0], m[col_i, 1], :] = out_vol[col_i]
                                if val_out["output_conf"] is not None:
                                    conf_vol[m[col_i, 0], m[col_i, 1], :] = val_out["output_conf"][0][col_i]
                            mask = masks[0].to(device=est_vol.device)
                            est_vol *= mask
                            conf_vol *= mask

                        else:
                            if len(out_vol.squeeze().shape)==1:
                                est_vol[m] = out_vol.reshape(-1)
                                if val_out["output_conf"] is not None:
                                    conf_vol = torch.ones_like(est_vol) * torch.nan
                                    prob_vol = torch.ones(est_vol.numel(), probs[0].shape[-1],
                                                          device=est_vol.device) * torch.nan
                                    conf_vol[m] = val_out["output_conf"][0]
                                    prob_vol[m] = probs[0]
                            else: # value, std
                                #not supported anymore
                                assert False
                                est_vol[m] = out_vol[:,0]
                                conf_vol = torch.ones_like(est_vol) * torch.nan
                                conf_vol[m] = out_vol[:, 1]
                time_net = time.time() - net_start_time
            else:
                time_net = 0
            assert len(est_vols)==1 ## TODO support inference with batch larger than 1


            gt_vol = val_volume.extinctions[0].squeeze()
            est_vols = est_vols.squeeze().reshape(gt_vol.shape)
            if val_out["output_conf"] is not None:

                conf_vol = conf_vol.squeeze().reshape(gt_vol.shape)
                prob_vol = prob_vol.reshape(*gt_vol.shape,-1)
            else:
                conf_vol = torch.empty(1)
                prob_vol = torch.empty(1)
            masks[0] = masks[0].squeeze().reshape(gt_vol.shape)
            est_vols[est_vols<0] = 0

            print(f'epsilon = {relative_error(ext_est=est_vols,ext_gt=gt_vol)}, relative_mass_error = {relative_mass_error(ext_est=est_vols,ext_gt=gt_vol)}, relative_voxel_error = {relative_voxel_error(ext_est=est_vols,ext_gt=gt_vol)}  L2 = {relative_squared_error(ext_est=est_vols,ext_gt=gt_vol)}, Npoints = {n_points_mask}')
            if False:
                xv, yv, zv = np.meshgrid(np.linspace(0, gt_vol.shape[0],
                                                     gt_vol.shape[0]),np.linspace(0, gt_vol.shape[1], gt_vol.shape[1]),
                                         np.linspace(0, gt_vol.shape[2], gt_vol.shape[2]))
                plt.scatter(gt_vol[masks[0]].ravel().cpu(), est_vols[masks[0]].ravel().cpu(),c=torch.log(conf_vol[masks[0]].ravel().cpu()))
                plt.colorbar()
                plt.plot([0,gt_vol[masks[0]].ravel().cpu().max()],[0,gt_vol[masks[0]].ravel().cpu().max()],'r')
                plt.xlabel('gt')
                plt.ylabel('est')
                plt.axis('square')
                plt.show()
                plt.scatter(torch.abs(gt_vol[est_vols>1].ravel().cpu() - est_vols[est_vols>1].ravel().cpu()), torch.log(conf_vol[est_vols>1].ravel().cpu()))
                plt.xlabel('|gt-est|')
                plt.ylabel('confidence')
                plt.show()


                if 0: #toy
                    x0, y0, z0 = (0.8, 0.8, 1.28)
                    x, y, z = np.meshgrid(np.linspace(0,31*0.05,32),np.linspace(0,31*0.05,32), np.linspace(0,63*0.04,64))
                    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
                    nx=ny=32
                    nz=64
                    mu = -1
                    sigma = 0.25
                    cloud_in = np.zeros((nx, ny, nz))
                    cloud_in[r <= 0.06] = 1
                    gt_vol_in = gt_vol[cloud_in==1]
                    est_vol_in = est_vols[cloud_in==1]
                    conf_vol_in = conf_vol[cloud_in==1]

                    cloud_mid = np.zeros((nx, ny, nz))
                    cloud_mid[(r > 0.06) * (r < 0.5)] = 1
                    gt_vol_mid = gt_vol[cloud_mid==1]
                    est_vol_mid = est_vols[cloud_mid==1]
                    conf_vol_mid = conf_vol[cloud_mid==1]

                    cloud_out = np.zeros((nx, ny, nz))
                    cloud_out[(r >= 0.5) * (r < 0.6)] = 1
                    gt_vol_out = gt_vol[cloud_out==1]
                    est_vol_out = est_vols[cloud_out==1]
                    conf_vol_out = conf_vol[cloud_out==1]

                    # sio.savemat('mask_toy_cloud2.mat',{'cloud_in':cloud_in,'cloud_mid':cloud_mid,'cloud_out':cloud_out})

            if cfg.save_results:
                val_image *= cfg.data.std
                val_image += cfg.data.mean
                sio.savemat(f'results_cloud_{val_i}.mat',{'gt':gt_vol.detach().cpu().numpy(),'est':est_vols.detach().cpu().numpy(),
                                                          'probs':prob_vol.cpu().numpy()})# 'images': val_image.detach().cpu().numpy(),

            # aggregate error statistics
            relative_err.append(relative_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            l2_err.append(relative_squared_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            relative_mass_err.append(relative_mass_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            abs_err.append(abs_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            relative_voxel_err.append(relative_voxel_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            confidence_list.append(conf_vol.cpu().numpy())
            est_list.append(est_vols.cpu().numpy())
            gt_list.append(gt_vol.cpu().numpy())
            mask_list.append(masks[0].cpu().numpy())

            batch_time_net.append(time_net)

            if cfg.show:
                show_scatter_plot(gt_vol,est_vols)
                show_scatter_plot_altitute(gt_vol,est_vols)
                volume_plot(gt_vol,est_vols)
            if writer:
                writer._iter = iteration
                writer._dataset = 'val'  # .format(val_i)
                if val_i in val_scatter_ind:
                    writer.monitor_scatter_plot(est_vols, gt_vol,ind=val_i)


    relative_err = np.array(relative_err)
    relative_mass_err =np.array(relative_mass_err)
    l2_err = np.array(l2_err)
    abs_err = np.array(abs_err)
    relative_voxel_err = np.array(relative_voxel_err)
    confidence_list= np.array(confidence_list)
    est_list= np.array(est_list)
    gt_list= np.array(gt_list)
    mask_list= np.array(mask_list)
    batch_time_net = np.array(batch_time_net)

    confidence_list2 = confidence_list.copy()
    weighted_relative_err = []
    # for est, gt, conf,mask in zip(est_list, gt_list, confidence_list2,mask_list):
    #     conf1 = np.zeros_like(gt)
    #     conf1[mask] = conf[mask]
    #     err = np.sum(conf * np.abs(est - gt)) / (np.sum(conf1) * np.sum(gt) + 1e-6) * np.sum(mask)
    #     weighted_relative_err.append(err)



    print(f'mean relative error {np.mean(relative_err)} with std of {np.std(relative_err)} for {(val_i + 1)} clouds')
    print(f'mean relative mass error {np.mean(relative_mass_err)} with std of {np.std(relative_mass_err)} for {(val_i + 1)} clouds')
    print(f'mean L2 error {np.mean(l2_err)} with std of {np.std(l2_err)} for {(val_i + 1)} clouds')
    print(f'mean abs error {np.mean(abs_err)} with std of {np.std(abs_err)} for {(val_i + 1)} clouds')
    print(f'mean relative voxel error {np.mean(relative_voxel_err)} with std of {np.std(relative_voxel_err)} for {(val_i + 1)} clouds')
    # print(f'Weighted relative error {np.mean(weighted_relative_err)} with std of {np.std(weighted_relative_err)} for {(val_i + 1)} clouds')

    print(f'Mean time = {np.mean(batch_time_net)} +- {np.std(batch_time_net)}')
    sio.savemat(f'numerical_results.mat', {'relative_err': relative_err, 'relative_mass_err': relative_mass_err,
                 'l2_err': l2_err, 'batch_time_net': batch_time_net})
    sio.savemat(f'raw_results.mat', {'est_list': est_list, 'gt_list': gt_list,
                 'mask_list': mask_list, 'confidence_list': confidence_list})


    masked = relative_err<2
    relative_err1 = relative_err[masked]
    relative_mass_err1 = relative_mass_err[masked]
    l2_err1 = l2_err[masked]

    print(f'mean relative error w/o outliers {np.mean(relative_err1)} with std of {np.std(relative_err1)} for {relative_err1.shape[0]} clouds')
    print(f'mean relative mass error w/o outliers {np.mean(relative_mass_err1)} with std of {np.std(relative_mass_err1)} for {relative_mass_err1.shape[0]} clouds')
    print(f'mean L2 error w/o outliers {np.mean(l2_err1)} with std of {np.std(l2_err1)} for {l2_err1.shape[0]} clouds')

if __name__ == "__main__":
    main()


