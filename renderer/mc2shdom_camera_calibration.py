import os
from renderer.mc_renderer import DiffRendererMC
from renderer.shdom_renderer import DiffRendererSHDOM
import hydra
from omegaconf import OmegaConf, DictConfig
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../", "configs")
import numpy as np
import torch
from dataloader.dataset import get_cloud_datasets, trivial_collate
from scene.cameras import PerspectiveCameras
from scene.volumes import Volumes

@hydra.main(config_path=CONFIG_DIR, config_name="ft_train", version_base='1.1')
def main(cfg: DictConfig):
    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0], '.hydra/config.yaml')

    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg = OmegaConf.merge(net_cfg, cfg)
    diff_renderer_mc = DiffRendererMC(cfg=cfg)
    diff_renderer_shdom = DiffRendererSHDOM(cfg=cfg)
    train_dataset, val_dataset = get_cloud_datasets(
        cfg=cfg
    )
    if torch.cuda.is_available() and cfg.debug == False:
        n_device = torch.cuda.device_count()
        cfg.gpu = 0 if n_device==1 else cfg.gpu
        device = f"cuda:{cfg.gpu}"
    else:

        device = "cpu"
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=trivial_collate,
    )
    for epoch in range(0, cfg.optimizer.max_epochs):
        for i, batch in enumerate(train_dataloader):
            images, extinction, grid, image_sizes, projection_matrix, camera_center, masks, cloud_path = batch  # [0]#.values()
            cameras = PerspectiveCameras(image_size=image_sizes,
                                         P=torch.tensor(np.array(projection_matrix), device=device).float(),
                                         camera_center=torch.tensor(np.array(camera_center), device=device).float(),
                                         device=device)
            # cloud_path = cloud_path[0].split('cloud_results_')[-1].split('.pkl')[0]
            volume = Volumes(torch.unsqueeze(torch.tensor(np.array(extinction), device=device).float(), 1), grid)

            masks = volume.extinctions[0] > volume._ext_thr


            # images = torch.tensor(np.array(images), device=device).float()
            images = np.array(images)
            import time
            t = time.time()
            diff_renderer_shdom.render(volume.extinctions[0, 0], masks[0], volume, images)
            elapsed_shdom = time.time() - t
            t = time.time()
            diff_renderer_mc.render(volume.extinctions[0, 0], masks[0], volume, images)
            elapsed_mc = time.time() - t
            print(f'runtime shdom={elapsed_shdom}, mc={elapsed_mc}')
            gt_images=diff_renderer_shdom.gt_images

            shdom_images=np.array(diff_renderer_shdom.images)
            mc_images=np.array(diff_renderer_mc.images)
            import matplotlib.pyplot as plt
            f, axarr = plt.subplots(3, images.shape[1], figsize=(16, 16))
            for ax, image in zip(axarr[0], shdom_images):
                ax.imshow(image)
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.axis('off')
            for ax, image in zip(axarr[1], mc_images):
                ax.imshow(image)
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.axis('off')
            for ax, image, image_mc in zip(axarr[2], shdom_images, mc_images):
                ax.imshow(image-image_mc)
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.axis('off')
            plt.tight_layout()
            plt.show()

            plt.scatter(shdom_images.ravel(), mc_images.ravel())
            plt.plot([0,shdom_images.max()],[0,shdom_images.max()],'--r')
            plt.show()
            print()

if __name__ == "__main__":
    main()
