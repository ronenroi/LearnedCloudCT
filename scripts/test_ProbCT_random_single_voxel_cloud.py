# This file contains the main script for VIP-CT and ProbCT and ProbCT models.
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
from ProbCT.CTnet import CTnet
from ProbCT.CTnetV2 import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
from ProbCT.util.plot_util import *
from ProbCT.util.visualization import SummaryWriter
import scipy.io as sio
from metrics.test_errors import *
from metrics.losses import *
from ProbCT import *
from scene.volumes import Volumes
from scene.cameras import PerspectiveCameras
import copy
import os, glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataloader.noise import SatelliteNoise
import scipy.stats, scipy

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
global probs

@hydra.main(config_path=CONFIG_DIR, config_name="vipctV2_test")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    def func_p_beta(beta):
        A = 0.75 * (scipy.stats.norm.pdf((beta - 42) * 0.2)) + 0.25 * (scipy.stats.norm.pdf((beta - 78) * 0.2))
        A /= 5
        return A


    data_root = '/wdata/roironen/Data'

    data_root = os.path.join(data_root, 'Toy_single_voxel_clouds', '10cameras_20m')
    data_train_path = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]

    images = []
    betas = []
    for path in data_train_path[:400]:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        images.append(data['images'][:])
        betas.append(data['ext'].ravel()[data['index']])

    data_train_path_uniform = [f for f in glob.glob(os.path.join(data_root, "train/LINESPACE_cloud_*.pkl"))]
    images_uniform = []
    betas_uniform = []
    for path in data_train_path_uniform:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        images_uniform.append(data['images'][:])
        betas_uniform.append(data['ext'].ravel()[data['index']])

    betas = np.array(betas)
    images = np.stack(images)
    args = np.argsort(betas)
    betas = betas[args]
    images = images[args]
    data_train_path = np.array(data_train_path)[args]

    y_ind = 50

    noise = SatelliteNoise(fullwell=13.5e3, bits=10, DARK_NOISE_std=13)
    y = noise.convert_radiance_to_graylevel(images[y_ind])

    path = data_train_path[y_ind]
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # y = noisy_images[y_ind]

    data2 = copy.deepcopy(data)
    if len(y.shape)==2:
        y=y[None]
    data2['images'] = y
    files = glob.glob('/wdata/roironen/Data/Toy_single_voxel_clouds/10cameras_20m/test/*')
    for f in files:
        os.remove(f)

    with open(path.replace("train", "test"), 'wb') as f:
        pickle.dump(data2, f, pickle.HIGHEST_PROTOCOL)

    hist = np.histogram(betas, bins=25)
    hist_dist = scipy.stats.rv_histogram(hist)

    # plt.imshow(noisy_images.std(0)[60:100, 40:80])
    # plt.colorbar()
    # plt.show()

    log_p_y_beta_list = []
    p_beta = []

    for cur_ind in range(len(betas_uniform)):
        clean_images = images_uniform[cur_ind][:,60:100, 40:80][None]
        noisy_images_curr = noise.convert_radiance_to_graylevel(np.repeat(clean_images, 300, axis=0))

        var = noisy_images_curr.std(0) ** 2
        # log_p_y_beta = -0.5*np.log(var.ravel()).sum() + \
        log_p_y_beta = -0.5*np.log(var.ravel()).sum() + (-0.5 * (y[:,60:100, 40:80].ravel() - clean_images.ravel()) ** 2 / (var.ravel())).sum()

        log_p_y_beta_list.append(log_p_y_beta)

        # p_beta.append(hist_dist.pdf(betas[cur_ind]))
        p_beta.append(func_p_beta(betas_uniform[cur_ind]))

    plt.plot(log_p_y_beta_list)
    plt.show()
    log_p_y_beta_list = scipy.ndimage.gaussian_filter1d(log_p_y_beta_list, 5)
    plt.plot(log_p_y_beta_list)
    plt.show()
    p_beta = np.array(p_beta)
    log_p_y_beta_list = np.array(log_p_y_beta_list)
    log_p_y_beta_list = log_p_y_beta_list - log_p_y_beta_list.max()
    p_y_beta = np.exp(log_p_y_beta_list)
    p_beta /= np.trapz(p_beta, betas_uniform)
    p_y_beta /= np.trapz(p_y_beta, betas_uniform)

    posterior = p_y_beta * p_beta
    posterior /= np.trapz(posterior, betas_uniform)
    # posterior = scipy.ndimage.gaussian_filter1d(posterior, 30)
    # posterior /= np.trapz(posterior, betas_uniform)



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


    model_pathes=['/wdata/roironen/Deploy/ProbCT/outputs/2023-05-25_Train_TOY3/16-14-06/checkpoints/cp_165000.pth',
                  '/wdata/roironen/Deploy/ProbCT/outputs/2023-05-26_Train_TOY3_3views/13-44-34/checkpoints/cp_310000.pth',
                  '/wdata/roironen/Deploy/ProbCT/outputs/2023-05-27_Train_TOY3_6views/10-17-35/checkpoints/cp_280000.pth',
                  '/wdata/roironen/Deploy/ProbCT/outputs/2023-05-26_Train_TOY3_10views/13-44-12/checkpoints/cp_180000.pth'
                  ]

    model_posteriors = []
    model_posteriors2 = []
    n_cams = [1,3,6,10]
    for model_path,n_cam in zip(model_pathes,n_cams):
        checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), model_path)


        resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
        net_cfg = OmegaConf.load(resume_cfg_path)
        cfg = OmegaConf.merge(net_cfg,cfg)
        cfg.data.n_cam = n_cam
        cfg.data.n_cam = n_cam
        cfg.backbone.n_sampling_nets = n_cam

        _, val_dataset = get_cloud_datasets(
            cfg=cfg
        )
        # Initialize VIP-CT or ProbCT models
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



        # Run the main training loop.
        for val_i, val_batch in enumerate(val_dataloader):
            val_image, extinction, grid, image_sizes, projection_matrix, camera_center, masks = val_batch
            val_image = torch.tensor(np.array(val_image), device=device).float()
            val_volume = Volumes(torch.unsqueeze(torch.tensor(np.array(extinction), device=device).float(), 1), grid)
            val_camera = PerspectiveCameras(image_size=image_sizes, P=torch.tensor(projection_matrix, device=device).float(),
                                            camera_center=torch.tensor(camera_center, device=device).float(), device=device)
            if model.val_mask_type == 'gt_mask':
                masks = val_volume.extinctions > 0 #val_volume._ext_thr
            else:
                masks = [torch.tensor(mask) if mask is not None else torch.ones(*extinction[0].shape,device=device, dtype=bool) for mask in masks]

            with torch.no_grad():

                val_out = model(
                    val_camera,
                    val_image,
                    val_volume,
                    masks
                )

                val_out["output"], val_out["output_conf"], probs = get_pred_and_conf_from_discrete(val_out["output"],
                                                                                            cfg.cross_entropy.min,
                                                                                            cfg.cross_entropy.max,
                                                                                            cfg.cross_entropy.bins,
                                                                                            pred_type=cfg.ct_net.pred_type,
                                                                                            conf_type=cfg.ct_net.conf_type)

            est_posterior = probs[0].cpu().numpy().T
            est_posterior2 = scipy.ndimage.gaussian_filter1d(est_posterior.ravel(), 2)
            # est_posterior2 = est_posterior2**8;
            est_posterior2 /= est_posterior2.sum()
            model_posteriors.append(est_posterior)
            model_posteriors2.append(est_posterior2)
    plt.plot(betas_uniform, p_beta)
    # plt.plot(betas_uniform, p_y_beta)
    plt.plot(betas_uniform, posterior)
    for est in model_posteriors:
        plt.plot(np.arange(0, 301), est)
    # plt.plot(np.arange(0, 301), est_posterior)
    # plt.plot(np.arange(0, 301), est_posterior)
    # plt.plot(np.arange(0, 301), est_posterior2)

    plt.xlim([0,100])
    plt.legend(['p_beta', 'posterior', 'Estimated - 1 view', 'Estimated - 3 views', 'Estimated - 6 views', 'Estimated - 10 views'])
    plt.show()

    plt.plot(betas_uniform, posterior)
    plt.hist(betas, bins=25, density=True)
    b = np.arange(20, 100)
    plt.plot(b, hist_dist.pdf(b))
    plt.plot(b, func_p_beta(b))
    plt.show()

    print(betas[y_ind])

    print(val_out["output"])
    print(betas_uniform[np.argmax(posterior)])

    print('DONE')


if __name__ == "__main__":
    main()


