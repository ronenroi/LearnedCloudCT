# This file contains the main script for VIP-CT training.
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
from dataloader.dataset import get_cloud_datasets, trivial_collate
from dataloader.airmspi_dataset import get_airmspi_datasets

from LearnedCloudCT.ProbCT.util.visualization import SummaryWriter
from LearnedCloudCT.ProbCT.CTnetV2 import *
from LearnedCloudCT.ProbCT.util.stats import Stats
from omegaconf import DictConfig
from metrics.test_errors import *
from metrics.losses import *
from ProbCT import *
from LearnedCloudCT.scene.volumes import Volumes
from LearnedCloudCT.scene.cameras import PerspectiveCameras

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

@hydra.main(config_path=CONFIG_DIR, config_name="vipctV2_train_airmspi")
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
    if "AirMSPI" in cfg.data.dataset_name:
        train_dataset, val_dataset = get_airmspi_datasets(
            cfg=cfg
        )
    else:
        train_dataset, val_dataset = get_cloud_datasets(
            cfg=cfg
        )

    # Initialize the CT model.
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
    checkpoint_resume_path =  os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(checkpoint_dir) > 0:
        # Make the root of the experiment directory.
        # checkpoint_dir = os.path.split(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume training if requested.
    if cfg.resume and os.path.isfile(checkpoint_resume_path):
        print(f"Resuming from checkpoint {checkpoint_resume_path}.")
        loaded_data = torch.load(checkpoint_resume_path, map_location=device)
        model.load_state_dict(loaded_data["model"])
        stats = pickle.loads(loaded_data["stats"])
        print(f"   => resuming from epoch {stats.epoch}.")
        optimizer_state_dict = loaded_data["optimizer"]
        start_epoch = stats.epoch

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
    if cfg.optimizer.ce_weight_zero>0:
        w = torch.ones(cfg.cross_entropy.bins, device=device)
        w[0] *= cfg.optimizer.ce_weight_zero
        CE.weight = w

    # err = torch.nn.L1Loss(reduction='sum')
    # Set the model to the training mode.
    model.train().float()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # Run the main training loop.
    iteration = -1

    if writer:
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        for i, batch in enumerate(train_dataloader):
            iteration += 1
            # lr_scheduler(None)
            if iteration % (cfg.stats_print_interval) == 0 and iteration > 0:
                stats.new_epoch()  # Init a new epoch.
            if iteration in cfg.optimizer.iter_steps:
                # Adjust the learning rate.
                lr_scheduler.step()
            if "AirMSPI" in cfg.data.dataset_name:
                images, extinction, grid, mapping, centers, masks = batch#[0]#.values()
                if images[0] is None:
                    continue
                cameras = AirMSPICameras(mapping=torch.tensor(mapping).float(),
                                         centers=torch.tensor(centers).float(),
                                             device=device)

            else:
                images, extinction, grid, image_sizes, projection_matrix, camera_center, masks = batch#[0]#.values()
                cameras = PerspectiveCameras(image_size=image_sizes,
                                             P=torch.tensor(projection_matrix, device=device).float(),
                                             camera_center=torch.tensor(camera_center, device=device).float(),
                                             device=device)


            images = torch.tensor(np.array(images), device=device).float()
            volume = Volumes(torch.unsqueeze(torch.tensor(extinction, device=device).float(),1), grid)

            masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]
            if model.mask_type == 'gt_mask':
                masks = volume.extinctions > volume._ext_thr
            if torch.sum(torch.tensor([(mask).sum() if mask is not None else mask for mask in masks])) == 0:
                continue
            optimizer.zero_grad()

            # Run the forward pass of the model.
            out = model(
                cameras,
                images,
                volume,
                masks
            )

            # The loss is a sum of coarse and fine MSEs
            if cfg.optimizer.loss == 'L2_relative_error_new':
                loss = [err(ext_est.squeeze(),ext_gt.squeeze())/(torch.norm(ext_gt.squeeze())**2 / (ext_gt.shape[0]+1e-2) / + 1e-2) for ext_est, ext_gt in zip(out["output"], out["volume"])]
            elif cfg.optimizer.loss == 'L2_relative_error':
                loss = [err(ext_est.squeeze(),ext_gt.squeeze())/(torch.norm(ext_gt.squeeze())+ 1e-2) for ext_est, ext_gt in zip(out["output"], out["volume"])]
            elif cfg.optimizer.loss == 'L2':
                loss = [err(ext_est.squeeze(),ext_gt.squeeze()) for ext_est, ext_gt in zip(out["output"], out["volume"])]
            elif cfg.optimizer.loss == 'L2_Weighted':
                weights = torch.ones(27,device=device)
                weights[:13] /= 5
                weights[14:] /= 5
                loss = [err(ext_est.squeeze()*weights,ext_gt.squeeze()*weights) for ext_est, ext_gt in zip(out["output"], out["volume"])]
            elif cfg.optimizer.loss == 'L1_relative_error':
                loss = [relative_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in zip(out["output"], out["volume"])]
            elif cfg.optimizer.loss == 'CE':
                loss = [CE(ext_est,
                           to_discrete(ext_gt, cfg.cross_entropy.min, cfg.cross_entropy.max, cfg.cross_entropy.bins))
                                                             for ext_est, ext_gt in zip(out["output"], out["volume"])]

            else:
                NotImplementedError
            loss = torch.mean(torch.stack(loss))
            # loss = torch.tensor(loss).mean()

            # Take the training step.
            try:
                loss.backward()
                optimizer.step()

            except:
                pass
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            with torch.no_grad():
                if cfg.ct_net.use_neighbours:
                    out["output"] = [ext_est.reshape(-1,3,3,3)[:,1,1,1] for ext_est in out["output"]]
                    out["volume"] = [ext_gt.reshape(-1, 3, 3, 3)[:, 1, 1, 1] for ext_gt in out["volume"]]

                if cfg.optimizer.loss == 'CE':
                    out["output"] = get_pred_from_discrete(out["output"], cfg.cross_entropy.min, cfg.cross_entropy.max, cfg.cross_entropy.bins)
                relative_err = [relative_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in zip(out["output"], out["volume"])]#torch.norm(out["output"] - out["volume"],p=1,dim=-1) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_err = torch.tensor(relative_err).mean()
                relative_mass_err = [relative_mass_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in zip(out["output"], out["volume"])]#(torch.norm(out["output"],p=1,dim=-1) - torch.norm(out["volume"],p=1,dim=-1)) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_mass_err = torch.tensor(relative_mass_err).mean()

            # Update stats with the current metrics.
            stats.update(
                {"loss": float(loss), "relative_error": float(relative_err), "lr":  lr_scheduler.get_last_lr()[0],#optimizer.param_groups[0]['lr'],#lr_scheduler.get_last_lr()[0]
                 "max_memory": float(round(torch.cuda.max_memory_allocated(device=device)/1e6))},
                stat_set="train",
            )

            if iteration % cfg.stats_print_interval == 0 and iteration > 0:
                stats.print(stat_set="train")
                if writer:
                    writer._iter = iteration
                    writer._dataset = 'train'
                    writer.monitor_loss(loss.item())
                    writer.monitor_scatterer_error(relative_mass_err, relative_err)
                    for ind in range(len(out["output"])):
                        writer.monitor_scatter_plot(out["output"][ind], out["volume"][ind],ind=ind)
                    # writer.monitor_images(images)

            # Validation
            if iteration % cfg.validation_iter_interval == 0 and iteration > 0:
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
                    model.eval()
                    with torch.no_grad():
                        val_out = model(
                            val_camera,
                            val_image,
                            val_volume,
                            masks
                        )
                        if cfg.optimizer.loss == 'CE':
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
                        est_vols = est_vols.squeeze()
                        if cfg.optimizer.loss == 'L2_relative_error':
                            loss_val += err(est_vols.squeeze(), gt_vol.squeeze()) / (torch.norm(gt_vol.squeeze())**2 / gt_vol.shape[0] + 1e-2)
                        elif cfg.optimizer.loss == 'L2':
                            loss_val += err(est_vols.squeeze(), gt_vol.squeeze())
                        elif cfg.optimizer.loss == 'L1_relative_error':
                            loss_val += relative_error(ext_est=est_vols,ext_gt=gt_vol)
                        # elif cfg.optimizer.loss == 'CE':
                        #     loss = [CE(ext_est,
                        #                to_discrete(ext_gt, cfg.cross_entropy.min, cfg.cross_entropy.max,
                        #                            cfg.cross_entropy.bins))
                        #             for ext_est, ext_gt in zip(val_out["output"], out["volume"])]
                        # loss_val += l1(val_out["output"], val_out["volume"]) / torch.sum(val_out["volume"]+1000)

                        relative_err += relative_error(ext_est=est_vols,ext_gt=gt_vol)#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
                        relative_mass_err += relative_mass_error(ext_est=est_vols,ext_gt=gt_vol)#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
                        if writer:
                            writer._iter = iteration
                            writer._dataset = 'val'  # .format(val_i)
                            if val_i in val_scatter_ind:
                                writer.monitor_scatter_plot(est_vols, gt_vol,ind=val_i)
                    # Update stats with the validation metrics.
                    stats.update({"loss": float(loss_val), "relative_error": float(relative_err)}, stat_set="val")

                loss_val /= (val_i + 1)
                relative_err /= (val_i + 1)
                relative_mass_err /= (val_i+1)

                if writer:
                    writer._iter = iteration
                    writer._dataset = 'val'#.format(val_i)
                    writer.monitor_loss(loss_val)
                    writer.monitor_scatterer_error(relative_mass_err, relative_err)
                    # writer.monitor_images(val_image)

                stats.print(stat_set="val")



                # Set the model back to train mode.
                model.train()

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
                    "optimizer": optimizer.state_dict(),
                    "stats": pickle.dumps(stats),
                }
                torch.save(data_to_store, curr_checkpoint_path)


if __name__ == "__main__":
    main()
