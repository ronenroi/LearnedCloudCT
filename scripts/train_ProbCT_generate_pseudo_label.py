# This file contains the main script for VIP-CT and ProbCT training.
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
# sys.path.insert(0, '/home/roironen/pyshdom-NN/projects')
import hydra
import numpy as np
import torch

from dataloader.dataset import get_cloud_datasets, trivial_collate
from ProbCT.util.visualization import SummaryWriter
from ProbCT.CTnetV2 import *
from ProbCT.CTnet import CTnet
from ProbCT.util.stats import Stats
from omegaconf import DictConfig
from metrics.test_errors import *
from metrics.losses import *
from ProbCT import *
from scene.volumes import Volumes
from scene.cameras import PerspectiveCameras
from renderer.shdom_renderer import DiffRendererSHDOM, LossSHDOM
# from shdom.shdom_nn import *
import matplotlib.pyplot as plt

# relative_error = lambda ext_est, ext_gt, eps=1e-6 : torch.norm(ext_est.view(-1) - ext_gt.view(-1),p=1) / (torch.norm(ext_gt.view(-1),p=1) + eps)
# mass_error = lambda ext_est, ext_gt, eps=1e-6 : (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
CE = torch.nn.CrossEntropyLoss(reduction='mean')

# def build_criterion(args):
#     weight = torch.ones(args.num_classes)
#     weight[args.eos_index] = args.eos_loss_coef
#     criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
#
#     device = torch.device('cuda')
#     criterion = criterion.to(device)
#     return criterion

@hydra.main(config_path=CONFIG_DIR, config_name="vipctV2_shdom_train")
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

    # Load the training/validation data.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # DATA_DIR = os.path.join(current_dir, "data")
    train_dataset, val_dataset = get_cloud_datasets(
        cfg=cfg
    )

    # Initialize the CT model.
    if cfg.version == 'V1':
        model = CTnet(cfg=cfg, n_cam=cfg.data.n_cam)
    else:
        model = CTnetV2(cfg=cfg, n_cam=cfg.data.n_cam)
    # Move the model to the relevant device.
    model.to(device)
    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0

    #
    log_dir = os.getcwd()
    writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(checkpoint_dir) > 0:
        # Make the root of the experiment directory.
        # checkpoint_dir = os.path.split(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume training if requested.
    if cfg.resume and os.path.isfile(checkpoint_resume_path):
        print(f"Resuming from checkpoint {checkpoint_resume_path}.")
        loaded_data = torch.load(checkpoint_resume_path,map_location=device)
        model.load_state_dict(loaded_data["model"])
        # stats = pickle.loads(loaded_data["stats"])
        # print(f"   => resuming from epoch {stats.epoch}.")
        # optimizer_state_dict = loaded_data["optimizer"]
        # start_epoch = stats.epoch

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.wd,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # Init the stats object.
    if stats is None:
        stats = Stats(
            ["loss", "relative_error", "lr", "max_memory", "sec/it"],
        )

    # Learning rate scheduler setup.

    # Following the original code, we use exponential decay of the
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch #/ cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=trivial_collate,
    )

    # The validation dataloader is just an endless stream of random samples.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
    )
    err = torch.nn.MSELoss()
    if cfg.optimizer.ce_weight_zero:
        w = torch.ones(cfg.cross_entropy.bins, device=device)
        w[0] /= 100
        CE.weight = w

    # err = torch.nn.L1Loss(reduction='sum')
    # Set the model to the training mode.
    model.train().float()
    diff_renderer_shdom = DiffRendererSHDOM(cfg=cfg)

    if cfg.ct_net.stop_encoder_grad:
        for name, param in model.named_parameters():
            # if 'decoder.decoder.2.mlp.7' in name or 'decoder.decoder.3' in name:
            if 'decoder' in name or '.bn' in name:
            # if '.bn' in name:

                param.requires_grad = True
            else:
                param.requires_grad = False
        if cfg.ct_net.encoder_mode == 'eval':
            model._image_encoder.eval()
            model.mlp_cam_center.eval()
            model.mlp_xyz.eval()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # Run the main training loop.
    iteration = 0

    if writer:
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]
    est_vols = []
    volumes = []
    mask_list = []
    images_list = []
    f_name_is_noise = 'no_noise' if (cfg.data.noise == 'None' or cfg.data.noise == False) else 'noise'
    for i, batch in enumerate(train_dataloader):
        file_name = train_dataloader.dataset.cloud_dir[i].replace('train',f'pseudo_train_{f_name_is_noise}')
        if os.path.exists(file_name):
            continue
        # lr_scheduler(None)
        if iteration % (cfg.stats_print_interval) == 0 and iteration > 0:
            stats.new_epoch()  # Init a new epoch.
        if iteration in cfg.optimizer.iter_steps:
            # Adjust the learning rate.
            lr_scheduler.step()

        images, extinction, grid, image_sizes, projection_matrix, camera_center, masks, _ = batch#[0]#.values()
        volume = Volumes(torch.unsqueeze(torch.tensor(extinction, device=device).float(),1), grid)

        if model.mask_type == 'gt_mask':
            masks = volume.extinctions[0] > volume._ext_thr
        masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]
        if torch.sum(torch.tensor([(mask).sum() if mask is not None else mask for mask in masks])) == 0:
            print('Empty mask skip')
            continue
        images = torch.tensor(np.array(images), device=device).float()
        cameras = PerspectiveCameras(image_size=image_sizes,P=torch.tensor(projection_matrix, device=device).float(),
                                     camera_center= torch.tensor(camera_center, device=device).float(), device=device)


        optimizer.zero_grad()

        # Run the forward pass of the model.
        out = model(
            cameras,
            images,
            volume,
            masks
        )
        if out["output"][0].shape[-1]==1:
            conf_vol = None
            mask_conf = masks[0]
        else:
            out["output"], out["output_conf"], probs = get_pred_and_conf_from_discrete(out["output"],
                                                                                cfg.cross_entropy.min,
                                                                                cfg.cross_entropy.max,
                                                                                cfg.cross_entropy.bins,
                                                                                pred_type=cfg.ct_net.pred_type,
                                                                                conf_type=cfg.ct_net.conf_type,
                                                                                prob_gain=cfg.ct_net.prob_gain)
            conf_vol = torch.zeros(volume.extinctions.numel(), device=volume.device)
            conf_vol[out['query_indices'][0]] = out["output_conf"][0]
            conf_vol = conf_vol.reshape(volume.extinctions.shape[2:]).to(device=masks[0].device)
            mask_conf = masks[0]  # * (conf_vol > 0.2) * (conf_vol < 0.8)

        est_vol = torch.zeros(volume.extinctions.numel(), device=volume.device)
        est_vol[out['query_indices'][0]] = out["output"][0].squeeze()
        est_vol = est_vol.reshape(volume.extinctions.shape[2:])
        images = images.cpu().numpy()
        loss = diff_renderer_shdom.render(est_vol, mask_conf, volume, images)

        cloud = {'images': diff_renderer_shdom.images,
                 'cameras_pos': camera_center[0],
                 'cameras_P': projection_matrix[0],
                 'grid': grid[0],
                 'observed_images': diff_renderer_shdom.gt_images,
                 'ext': est_vol.detach().cpu().numpy(),
                 'mask': masks[0].detach().cpu().numpy(),
                 'model': checkpoint_resume_path
                 }

        with open(file_name, 'wb') as outfile:
            pickle.dump(cloud, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'saved at {file_name}')



if __name__ == "__main__":
    main()
