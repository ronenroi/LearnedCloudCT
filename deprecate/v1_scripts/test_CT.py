# This file contains the main script for VIP-CT evaluation.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite our papers described in the readme file.:
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
import hydra
import numpy as np
import torch
from LearnedCloudCT.dataloader.dataset import get_cloud_datasets, trivial_collate
from LearnedCloudCT.ProbCT.CTnet import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
from LearnedCloudCT.ProbCT.util.plot_util import *
from LearnedCloudCT.ProbCT.util.visualization import SummaryWriter
import scipy.io as sio
from LearnedCloudCT.scene.volumes import Volumes


# error criteria
relative_error = lambda ext_est, ext_gt, eps=1e-6 : torch.norm(ext_est.view(-1) - ext_gt.view(-1),p=1) / (torch.norm(ext_gt.view(-1),p=1) + eps)
L2_error = lambda ext_est, ext_gt, eps=1e-6 : torch.sum((ext_est.view(-1) - ext_gt.view(-1))**2) / (torch.sum((ext_gt.view(-1))**2) + eps)
mass_error = lambda ext_est, ext_gt, eps=1e-6 : (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../configs")

@hydra.main(config_path=CONFIG_DIR, config_name="vip-ct_test")
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
    model = CTnet(cfg=cfg, n_cam=cfg.data.n_cam)

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
    model.eval().float()

    iteration = -1
    if writer:
        # show scatter plot of 5 clouds
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]

    relative_err = []
    l2_err = []
    relative_mass_err = []
    batch_time_net = []
    val_i = 0

    # Run the main training loop.
    for val_i, val_batch in enumerate(val_dataloader):
        val_image, extinction, grid, image_sizes, projection_matrix, camera_center, masks = val_batch
        val_image = torch.tensor(val_image, device=device).float()
        val_volume = Volumes(torch.unsqueeze(torch.tensor(extinction, device=device).float(), 1), grid)
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

            # don't make inference on empty/small clouds
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
                        est_vol[m]=out_vol.squeeze(1)
                time_net = time.time() - net_start_time
            else:
                time_net = 0
            assert len(est_vols)==1 ## TODO support inference with batch larger than 1


            gt_vol = val_volume.extinctions[0].squeeze()
            est_vols = est_vols.squeeze().reshape(gt_vol.shape)
            est_vols[est_vols<0] = 0

            print(f'epsilon = {relative_error(ext_est=est_vols,ext_gt=gt_vol)}, L2 = {L2_error(ext_est=est_vols,ext_gt=gt_vol)}, Npoints = {n_points_mask}')

            if cfg.save_results:
                val_image *= cfg.data.std
                val_image += cfg.data.mean
                sio.savemat(f'results_cloud_{val_i}.mat',{'gt':gt_vol.detach().cpu().numpy(),'est':est_vols.detach().cpu().numpy(), 'images': val_image.detach().cpu().numpy()})

            # aggregate error statistics
            relative_err.append(relative_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            l2_err.append(L2_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
            relative_mass_err.append(mass_error(ext_est=est_vols,ext_gt=gt_vol).detach().cpu().numpy())
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
    batch_time_net = np.array(batch_time_net)
    print(f'mean relative error {np.mean(relative_err)} with std of {np.std(relative_err)} for {(val_i + 1)} clouds')
    print(f'mean relative mass error {np.mean(relative_mass_err)} with std of {np.std(relative_mass_err)} for {(val_i + 1)} clouds')
    print(f'mean L2 error {np.mean(l2_err)} with std of {np.std(l2_err)} for {(val_i + 1)} clouds')
    print(f'Mean time = {np.mean(batch_time_net)} +- {np.std(batch_time_net)}')
    sio.savemat(f'numerical_results.mat', {'relative_err': relative_err, 'relative_mass_err': relative_mass_err,
                 'l2_err': l2_err, 'batch_time_net': batch_time_net})

    masked = relative_err<2
    relative_err1 = relative_err[masked]
    relative_mass_err1 = relative_mass_err[masked]
    l2_err1 = l2_err[masked]

    print(f'mean relative error w/o outliers {np.mean(relative_err1)} with std of {np.std(relative_err1)} for {relative_err1.shape[0]} clouds')
    print(f'mean relative mass error w/o outliers {np.mean(relative_mass_err1)} with std of {np.std(relative_mass_err1)} for {relative_mass_err1.shape[0]} clouds')
    print(f'mean L2 error w/o outliers {np.mean(l2_err1)} with std of {np.std(l2_err1)} for {l2_err1.shape[0]} clouds')

if __name__ == "__main__":
    main()


