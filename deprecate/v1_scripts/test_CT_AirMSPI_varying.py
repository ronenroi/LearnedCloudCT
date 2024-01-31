# This file contains the main script for VIP-CT and ProbCT evaluation on AirMSPI data.
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

import os, time
import pickle
import warnings
# import sys
# sys.path.insert(0, '/home/roironen/pytorch3d/projects/')
import hydra
import numpy as np
from ProbCT.CTnet import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
from scene.cameras import AirMSPICameras
import scipy.io as sio
from ProbCT.scene.volumes import Volumes

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../", "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="vip-ct_test_airmspi")
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

    # Initialize ProbCT model
    model = CTnetAirMSPI(cfg=cfg, n_cam=cfg.data.n_cam)

    # Load model
    assert os.path.isfile(checkpoint_resume_path)
    print(f"Resuming from checkpoint {checkpoint_resume_path}.")
    loaded_data = torch.load(checkpoint_resume_path, map_location=device)
    model.load_state_dict(loaded_data["model"])
    model.to(device)
    model.eval().float()


    batch_time_net = []
    val_image = sio.loadmat('/wdata/yaelsc/AirMSPI_raw_data/raw_data/croped_airmspi_9images_for_Roi.mat')['croped_airmspi_images']
    if cfg.data.n_cam != 9:
        val_image = np.delete(val_image, cfg.data.drop_index, 0)
    val_image = torch.tensor(val_image,device=device).float()[None]

    # masks = sio.loadmat('/wdata/yaelsc/AirMSPI_raw_data/raw_data/mask_72x72x32_vox50x50x40m.mat')['mask']
    masks = sio.loadmat('/wdata/roironen/Data/mask_72x72x32_vox50x50x40mROI.mat')['mask']
    # mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
    # mapping_path = '/wdata/yaelsc/AirMSPI_raw_data/raw_data/voxel_pixel_list72x72x32_BOMEX_img350x350.pkl'
    # mapping_path = '/wdata/roironen/Data/voxel_pixel_list72x72x32_BOMEX_img350x350_processed.pkl'
    # with open(mapping_path, 'rb') as f:
    #     mapping = pickle.load(f)
    # images_mapping_list = sio.loadmat(mapping_path)['map']
    # pixel_center_path = '/wdata/roironen/Data/AirMSPI-Varying/test/20130206_202754Z_NorthPacificOcean-32N123W_vadim_measurement.mat'
    # image_size = [350, 350]
    # with open(mapping_path, 'rb') as f:
    #     mapping = pickle.load(f)
    # images_mapping_list = []
    # pixel_centers_list = []
    # pixel_centers = sio.loadmat(pixel_center_path)['xpc']
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
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_images_mapping_lists72x72x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_list, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_pixel_centers_lists72x72x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_list, f, pickle.HIGHEST_PROTOCOL)

    with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_images_mapping_lists72x72x32_BOMEX_img350x350.pkl', 'rb') as f:
        images_mapping_list = pickle.load(f)
    with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_pixel_centers_lists72x72x32_BOMEX_img350x350.pkl', 'rb') as f:
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
    gx = np.linspace(0,0.05 * 72,72, dtype=np.float32)
    gy = np.linspace(0, 0.05 * 72, 72, dtype=np.float32)
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


