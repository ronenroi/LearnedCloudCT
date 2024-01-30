# This file contains the main script for VIP-CT and ProbCT training.
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


import collections
import os, time
import pickle
import warnings
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
from renderer.at3d_renderer import DiffRendererAT3D
# from shdom.shdom_nn import *
import matplotlib.pyplot as plt


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../", "configs")
CE = torch.nn.CrossEntropyLoss(reduction='mean')

# def build_criterion(args):
#     weight = torch.ones(args.num_classes)
#     weight[args.eos_index] = args.eos_loss_coef
#     criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
#
#     device = torch.device('cuda')
#     criterion = criterion.to(device)
#     return criterion

@hydra.main(config_path=CONFIG_DIR, config_name="ft_train")
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
            ["loss", "relative_error", "mass_error", "lr", "max_memory", "sec/it"],
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
    diff_renderer_at3d = DiffRendererAT3D(cfg=cfg)

    if cfg.ct_net.stop_encoder_grad:
        for name, param in model.named_parameters():
            # if 'decoder.decoder.2.mlp.7' in name or 'decoder.decoder.3' in name:
            if 'decoder' in name:
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
    est_vol_list= []
    masks_conf = []
    images_list =[]
    relative_err_tot = 0
    relative_mass_err_tot = 0
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        for i, batch in enumerate(train_dataloader):
            # lr_scheduler(None)
            if iteration % (cfg.stats_print_interval) == 0 and iteration > 0:
                stats.new_epoch()  # Init a new epoch.
            if iteration in cfg.optimizer.iter_steps:
                # Adjust the learning rate.
                lr_scheduler.step()

            images, extinction, grid, image_sizes, projection_matrix, camera_center, masks = batch#[0]#.values()
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
                mask_conf = masks[0] # * (conf_vol < 0.2) #* (conf_vol < 0.8)
            with torch.no_grad():
                relative_err = [relative_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in zip(out["output"], out["volume"])]#torch.norm(out["output"] - out["volume"],p=1,dim=-1) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_err_tot += torch.tensor(relative_err).mean()
                relative_mass_err = [relative_mass_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in zip(out["output"], out["volume"])]#(torch.norm(out["output"],p=1,dim=-1) - torch.norm(out["volume"],p=1,dim=-1)) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_mass_err_tot += torch.tensor(relative_mass_err).mean()
            est_vol = torch.zeros(volume.extinctions.numel(), device=volume.device)
            est_vol[out['query_indices'][0]] = out["output"][0].squeeze()
            est_vol = est_vol.reshape(volume.extinctions.shape[2:])

            mask_conf = mask_conf #* (est_vol > 1).to(device=mask_conf.device)

            # gt_vol = extinction[0]
            # gt_mask = gt_vol>0
            # M = (masks[0].detach().cpu().numpy() + gt_mask).astype(bool)
            # if conf_vol is not None:
            #     plt.scatter(gt_vol[M].ravel(), est_vol[M].ravel().detach().cpu(),
            #                 c=conf_vol[M].ravel().cpu().detach())
            # else:
            #     plt.scatter(gt_vol[M].ravel(), est_vol[M].ravel().detach().cpu())
            # plt.colorbar()
            # plt.plot([0, gt_vol[M].ravel().max()], [0, gt_vol[M].ravel().max()], 'r')
            # plt.xlabel('gt')
            # plt.ylabel('est')
            # plt.axis('square')
            # plt.show()
            # if conf_vol is not None:
            #     plt.scatter(np.abs(gt_vol[M].ravel() - est_vol[M].ravel().cpu().detach().numpy()),
            #                 conf_vol[M].ravel().cpu().detach())
            #     plt.xlabel('|gt-est|')
            #     plt.ylabel('confidence')
            #     plt.show()

            print(mask_conf.sum())
            images = images.cpu().numpy()
            est_vol_list.append(est_vol)
            masks_conf.append(mask_conf)
            images_list.append(images)

            # if mask_conf.sum()>5000:
            #     print('skip')
            #     continue


            # print(est_vol[extinction[0]>0].mean().item())
            if len(est_vol_list) == cfg.shdom.n_clouds:
                loss = diff_renderer_at3d.parallel_render(est_vol_list, masks_conf, images_list)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.clip)
                if cfg.optimizer.loss_thr<diff_renderer_at3d.loss:
                    print("Images are too inconsistent, skip gradient update for stability")
                    continue
                loss.backward()
                skip=False
                for param in model.decoder.parameters():
                    if param.requires_grad and not torch.all(torch.isfinite(param.grad)):
                        skip = True
                        continue
                if skip:
                    print("invalid gradients")
                    est_vol_list = []
                    masks_conf = []
                    images_list = []
                    continue
                # Take the training step.
                iteration += 1
                optimizer.step()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                # Update stats with the current metrics.
                relative_err_tot /= len(est_vol_list)
                relative_mass_err_tot /= len(est_vol_list)
                stats.update(
                    {"loss": float(loss), "relative_error": float(relative_err_tot), "mass_error": float(relative_mass_err_tot),  "lr":lr_scheduler.get_last_lr()[0],#optimizer.param_groups[0]['lr'],#lr_scheduler.get_last_lr()[0]
                     "max_memory": float(round(torch.cuda.max_memory_allocated(device=device)/1e6))},
                    stat_set="train",
                )

                if iteration % cfg.stats_print_interval == 0 and iteration > 0:
                    stats.print(stat_set="train")
                    if writer:
                        writer._iter = iteration
                        writer._dataset = 'train'
                        writer.monitor_loss(loss.item())
                        writer.monitor_scatterer_error(relative_mass_err_tot, relative_err_tot)
                        for ind in range(len(out["output"])):
                            writer.monitor_scatter_plot(out["output"][ind], out["volume"][ind],ind=ind)
                            writer.monitor_images(diff_renderer_at3d.gt_images,np.array(diff_renderer_at3d.images))
                est_vol_list = []
                masks_conf = []
                images_list = []
                relative_err_tot = 0
                relative_mass_err_tot = 0
            # with torch.cuda.device(device=device):
            #     torch.cuda.empty_cache()
            # Validation
            # for mode in range(2)
                if iteration % cfg.validation_iter_interval == 0 and iteration > 0:
                    optimizer.zero_grad()
                    del images
                    loss_val = 0
                    relative_err= 0
                    relative_mass_err = 0
                    val_i = 0
                    for val_i, val_batch in enumerate(val_dataloader):

                    # val_batch = next(val_dataloader.__iter__())

                        val_image, extinction, grid, image_sizes, projection_matrix, camera_center, masks = val_batch#[0]#.values()
                        val_image = torch.tensor(val_image, device=device).float()
                        val_volume = Volumes(torch.unsqueeze(torch.tensor(extinction, device=device).float(), 1), grid)
                        val_camera = PerspectiveCameras(image_size=image_sizes,P=torch.tensor(projection_matrix, device=device).float(),
                                             camera_center= torch.tensor(camera_center, device=device).float(), device=device)
                        masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]
                        if model.val_mask_type == 'gt_mask':
                            masks = val_volume.extinctions > val_volume._ext_thr
                        if torch.sum(torch.tensor([(mask).sum() if mask is not None else mask for mask in masks])) == 0:
                            continue
                    # Activate eval mode of the model (lets us do a full rendering pass).
                    #     model.eval()
                        with torch.no_grad():
                            val_out = model(
                                val_camera,
                                val_image,
                                val_volume,
                                masks
                            )
                            if val_out["output"][0].shape[-1] > 1:
                                val_out["output"] = get_pred_from_discrete(val_out["output"], cfg.cross_entropy.min,
                                                                  cfg.cross_entropy.max, cfg.cross_entropy.bins)

                            est_vols = torch.zeros(torch.squeeze(val_volume.extinctions,1).shape, device=val_volume.device)
                            if cfg.ct_net.use_neighbours:
                                val_out["output"] = [ext_est.reshape(-1, 3, 3, 3)[:, 1, 1, 1].unsqueeze(-1) for ext_est in val_out["output"]]
                            if val_out['query_indices'] is None:
                                for i, (out_vol, m) in enumerate(zip(val_out["output"], masks)):
                                    est_vols[i][m.squeeze(0)] = out_vol.squeeze(1)
                            else:
                                for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                                    est_vol.reshape(-1)[m] = out_vol.reshape(-1)  # .reshape(m.shape)[m]
                            assert len(est_vols)==1 ##TODO support validation with batch larger than 1
                            gt_vol = val_volume.extinctions[0].squeeze()
                            est_vol = est_vols.squeeze()
                            if cfg.optimizer.loss == 'L2_relative_error':
                                loss_val += err(est_vol.squeeze(), gt_vol.squeeze()) / (torch.norm(gt_vol.squeeze())**2 / gt_vol.shape[0] + 1e-2)
                            elif cfg.optimizer.loss == 'L2':
                                loss_val += err(est_vol.squeeze(), gt_vol.squeeze())
                            elif cfg.optimizer.loss == 'L1_relative_error':
                                loss_val += relative_error(ext_est=est_vol,ext_gt=gt_vol)
                            # elif cfg.optimizer.loss == 'CE':
                            #     loss = [CE(ext_est,
                            #                to_discrete(ext_gt, cfg.cross_entropy.min, cfg.cross_entropy.max,
                            #                            cfg.cross_entropy.bins))
                            #             for ext_est, ext_gt in zip(val_out["output"], out["volume"])]
                            # loss_val += l1(val_out["output"], val_out["volume"]) / torch.sum(val_out["volume"]+1000)

                            relative_err += relative_error(ext_est=est_vol,ext_gt=gt_vol)#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
                            relative_mass_err += relative_mass_error(ext_est=est_vol,ext_gt=gt_vol)#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
                            if writer:
                                writer._iter = iteration
                                writer._dataset = 'val'  # .format(val_i)
                                if val_i in val_scatter_ind:
                                    writer.monitor_scatter_plot(est_vol, gt_vol,ind=val_i)
                        # Update stats with the validation metrics.

                    loss_val /= (val_i + 1)
                    relative_err /= (val_i + 1)
                    relative_mass_err /= (val_i+1)
                    print(f'[val] | epoch {epoch} | it {iteration} |  loss: {loss_val} | relative_err: {relative_err} |relative_mass_err: {relative_mass_err}')
                    stats.update({"loss": float(loss_val), "relative_error": float(relative_err)}, stat_set="val")

                    if writer:
                        writer._iter = iteration
                        writer._dataset = 'val'#.format(val_i)
                        writer.monitor_loss(loss_val)
                        writer.monitor_scatterer_error(relative_mass_err, relative_err)
                        # writer.monitor_images(val_image)

                    stats.print(stat_set="val")
                    # Set the model back to train mode.
                    del val_camera, val_image, val_volume, masks
                    # with torch.cuda.device(device=device):
                    #     torch.cuda.empty_cache()
                    # model.decoder.train()

                    # Checkpoint.
                if (
                    iteration % cfg.checkpoint_iteration_interval == 0
                    and len(checkpoint_dir) > 0
                    and iteration > 0
                ):
                    curr_checkpoint_path = os.path.join(checkpoint_dir,f'cp_{iteration}.pth')
                    print(f"Storing checkpoint {curr_checkpoint_path}.")
                    data_to_store = {
                        "model": model.state_dict(),
                    }
                    torch.save(data_to_store, curr_checkpoint_path)


if __name__ == "__main__":
    main()
