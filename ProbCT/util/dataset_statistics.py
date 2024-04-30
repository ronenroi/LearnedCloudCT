import copy
import pickle
import matplotlib.pyplot as plt
import glob, os
import scipy.io as sio
import numpy as np
from sklearn.manifold import TSNE
import os, time
import warnings
warnings.filterwarnings('ignore')
import hydra
from hydra import compose, initialize
import numpy as np
from omegaconf import OmegaConf

from dataloader.dataset import get_cloud_datasets, trivial_collate
from ProbCT.CTnetV2 import *
from ProbCT.util.discritize import get_pred_and_conf_from_discrete
from scene.volumes import Volumes
from metrics.test_errors import *
import scipy.io as sio

import matplotlib.pyplot as plt

hydra.core.global_hydra.GlobalHydra.instance().clear()
CONFIG_DIR = os.path.join("../", "../", "configs")

# Load ProbCT model

def load_model(MODEL_DIR):
    with initialize(version_base='1.1', config_path=CONFIG_DIR, job_name="test_app"):
        cfg = compose(config_name="test")
    # Set the relevant seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    resume_cfg_path = os.path.join(MODEL_DIR.split('/checkpoints')[0], '.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg = OmegaConf.merge(net_cfg, cfg)
    cfg.checkpoint_resume_path = os.path.join(MODEL_DIR, "model.pth")

    # Initialize ProbCT model
    model = CTnetV2(cfg=cfg, n_cam=cfg.data.n_cam)

    # Load model
    print(f"Resuming from checkpoint {cfg.checkpoint_resume_path}.")
    loaded_data = torch.load(cfg.checkpoint_resume_path, map_location='cpu')
    model.load_state_dict(loaded_data["model"])
    # Device on which to run
    if torch.cuda.is_available() and cfg.debug == False:
        n_device = torch.cuda.device_count()
        cfg.gpu = 0 if n_device == 1 else cfg.gpu
        device = f"cuda:{cfg.gpu}"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"

    # Set the model to eval mode.
    if cfg.mode == 'eval':
        model.eval().float()
    else:
        model.float()
    return model, cfg, device






def get_data(cfg, device, dataset_name):
    # Load inference data
    # DATA_DIR = os.path.join("../../", "data")
    cfg.data.dataset_name = dataset_name

    _, val_dataset = get_cloud_datasets(cfg=cfg)
    cfg.ct_net.conf_type = 'entropy'
    cfg.ct_net.decoder_batchify = True
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
    )



    # n_points_mask =  np.array(masks).sum()
    # gt_vol = val_volume.extinctions[0].squeeze()
    return val_dataloader


def probct_inference(model, val_dataloader, device, cfg):
    pred_clouds=[]
    for val_i, val_batch in enumerate(val_dataloader):
        val_image, extinction, grid, image_sizes, projection_matrix, camera_center, masks, _ = val_batch
        val_image = torch.tensor(np.array(val_image), device=device).float()
        val_volume = Volumes(torch.unsqueeze(torch.tensor(np.array(extinction), device=device).float(), 1), grid)
        val_camera = PerspectiveCameras(image_size=image_sizes, P=torch.tensor(projection_matrix, device=device).float(),
                                        camera_center=torch.tensor(camera_center, device=device).float(), device=device)
        masks = torch.tensor(masks)
        with torch.no_grad():
            est_vols = torch.zeros(val_volume.extinctions.numel(), device=val_volume.device).reshape(
                val_volume.extinctions.shape[0], -1)
            n_points_mask = torch.sum(torch.stack(masks) * 1.0) if isinstance(masks, list) else masks.sum()
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
                if val_out["output"][0].shape[-1] > 1:
                    val_out["output"], val_out["output_conf"], probs = get_pred_and_conf_from_discrete(val_out["output"],
                                                                                                       cfg.cross_entropy.min,
                                                                                                       cfg.cross_entropy.max,
                                                                                                       cfg.cross_entropy.bins,
                                                                                                       pred_type=cfg.ct_net.pred_type,
                                                                                                       conf_type=cfg.ct_net.conf_type)
                for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                    est_vol[m] = out_vol.reshape(-1)
                    conf_vol = torch.ones_like(est_vol) * torch.nan
                    prob_vol = torch.ones(est_vol.numel(), probs[0].shape[-1],
                                          device=est_vol.device) * torch.nan
                    conf_vol[m] = val_out["output_conf"][0]
                    prob_vol[m] = probs[0]


                gt_vol = val_volume.extinctions[0].squeeze()
                est_vols = est_vols.squeeze().reshape(gt_vol.shape)
                if val_out["output_conf"] is not None:
                    conf_vol = conf_vol.squeeze().reshape(gt_vol.shape)
                    prob_vol = prob_vol.reshape(*gt_vol.shape, -1)
                else:
                    conf_vol = torch.empty(1)
                    prob_vol = torch.empty(1)
                if est_vols.shape==(32,32,64):
                    pred_clouds.append(est_vols)
        if len(pred_clouds)==200:
            break
    return pred_clouds


def cloud_tsne():
    datasets = [
        "/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/test",
        '/wdata/roironen/Data/BOMEX_128x128x100_5000CCN_50m_micro_256/10cameras_20m/test',# "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/test",
        "/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/test",
        "/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/test",
        # "/wdata/roironen/Data/DYCOMS_RF02_500CCN_64x64x159_50m/10cameras_20m/test",
        # "/wdata/roironen/Data/DYCOMS_RF02_50CCN_64x64x159_50m/10cameras_20m/test",

    ]
    exts = []
    grids = []
    legend = ["BOMEX50CCN", "BOMEX500CCN", "CASS600CCN", "HAWAII2000CCN","DYCOMS_RF02_500CCN","DYCOMS_RF02_50CCN"][:len(datasets)]
    names = []
    for dataset, name in zip(datasets,legend):
        data_paths = [f for f in glob.glob(os.path.join(dataset, "cloud*.pkl"))]
        ext = []
        grid=None
        for cloud_path in data_paths:
            with open(cloud_path, 'rb') as f:
                A = pickle.load(f)['ext']
                if len(ext)==0 or A.shape==ext[-1].shape:
                    ext.append(A)
                if grid is None:
                    with open(cloud_path, 'rb') as f:
                        grid = pickle.load(f)['grid'][-1]
                        grids.append(grid)
        clouds = np.array(ext)
        if not np.isclose(grid[1]-grid[0],0.04):
            clouds=clouds[:,:,:,::2][...,:64]
        exts.append(np.log(np.abs(np.fft.fftn(clouds[:200], axes=(1,2,3)))))
        names+=[name]*200

    # fig, axs = plt.subplots(2, 3)
    # axs = axs.ravel()


    X=np.vstack(exts)
    X=X.reshape(X.shape[0],-1)
    


    

    print()

    MODEL_DIR = os.path.join("models", "Test_BOMEX500", "Trained_BOMEX500")
    model, cfg, device = load_model(MODEL_DIR=MODEL_DIR)
    model.to(device)
    val_dataloader = get_data(cfg, device=device, dataset_name='BOMEX_5000CCN_new_10cameras_20m')
    pred_clouds = probct_inference(model, val_dataloader, device, cfg)
    pred_clouds = np.log(np.abs(np.fft.fftn(torch.stack(pred_clouds).cpu().numpy(),axes=(1,2,3))))
    
    pred_clouds = pred_clouds.reshape(pred_clouds.shape[0], -1)
    pred_clouds_bomex = pred_clouds[np.isfinite(pred_clouds.sum(-1))]

    MODEL_DIR = os.path.join("models", "Test_CASS600", "Trained_CASS600")
    model, cfg, device = load_model(MODEL_DIR=MODEL_DIR)
    model.to(device)
    val_dataloader = get_data(cfg, device=device, dataset_name='CASS_600CCN_roiprocess_10cameras_20m')
    pred_clouds = probct_inference(model, val_dataloader, device, cfg)
    pred_clouds = np.log(np.abs(np.fft.fftn(torch.stack(pred_clouds).cpu().numpy(), axes=(1, 2, 3))))

    pred_clouds = pred_clouds.reshape(pred_clouds.shape[0], -1)
    pred_clouds_cass = pred_clouds[np.isfinite(pred_clouds.sum(-1))]

    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12)
    data = tsne.fit_transform(np.concatenate((X,pred_clouds_bomex,pred_clouds_cass)))
    X_2d = data[:800]
    pred_2d_bomex = data[800:1000]
    pred_2d_cass = data[1000:]
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c'

    target_names = np.array(names)
    for c, label in zip(colors, legend):
        plt.scatter(X_2d[target_names == label, 0], X_2d[target_names == label, 1], c=c, label=label)
    
    plt.scatter(pred_2d_bomex[:, 0], pred_2d_bomex[:, 1], c=colors[1], label='BOMEX500CCN', marker='x')
    plt.scatter(pred_2d_cass[:, 0], pred_2d_cass[:, 1], c=colors[2], label='CASS600CCN', marker='x')
    plt.legend()
    plt.show()
    print()
    # for e, ax, l in zip(exts,axs,legend):
    #     clouds = np.array(e)
    #     clouds = clouds[clouds>0]
    #     ax.hist(clouds,100,density=True)
    #     # ax.title(l)
    #     ax.set(xlabel='Voxel extinction value [1/km]', ylabel='Probability',title=l)
    # fig.tight_layout()
    # plt.show()


def cloud_histograms():
    datasets = [
        "/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/test",
        "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/test",
        "/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/test",
        "/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/test",
        # "/wdata/roironen/Data/DYCOMS_RF02_500CCN_64x64x159_50m/10cameras_20m/test",
        # "/wdata/roironen/Data/DYCOMS_RF02_50CCN_64x64x159_50m/10cameras_20m/test",

    ]
    exts = []
    for dataset in datasets:
        data_paths = [f for f in glob.glob(os.path.join(dataset, "cloud*.pkl"))]
        ext = []
        for cloud_path in data_paths:
            with open(cloud_path, 'rb') as f:
                ext.append(pickle.load(f)['ext'])
        clouds = np.array(ext)
        exts.append(clouds[clouds>0])

    # fig, axs = plt.subplots(2, 3)
    # axs = axs.ravel()
    legend = ["BOMEX50CCN", "BOMEX500CCN", "CASS600CCN", "HAWAII2000CCN","DYCOMS_RF02_500CCN","DYCOMS_RF02_50CCN"]
    # for e, ax, l in zip(exts,axs,legend):
    #     clouds = np.array(e)
    #     clouds = clouds[clouds>0]
    #     ax.hist(clouds,100,density=True)
    #     # ax.title(l)
    #     ax.set(xlabel='Voxel extinction value [1/km]', ylabel='Probability',title=l)
    # fig.tight_layout()
    # plt.show()

    fig, axs = plt.subplots(1, 1)
    for e in exts:
        clouds = np.array(e)
        clouds = clouds[clouds>0]
        plt.hist(clouds, 500, density=True, histtype='step', stacked=True, fill=False)
        # plt.hist(clouds,100,density=True,alpha=0.5)
    plt.legend(legend)
    plt.xlabel('Voxel extinction value [1/km]')
    plt.ylabel('Probability')
    # plt.xlim([0,500])
    plt.xscale("log")

    fig.tight_layout()
    image_format = 'svg'  # e.g .png, .svg, etc.
    image_name = '/wdata/roironen/Deploy/ProbCT/data_hist.svg'

    fig.savefig(image_name, format=image_format, dpi=1200)
    plt.show()
    print()
def get_cloud_top():
    datasets = [
        # "/wdata_visl/NEW_BOMEX/processed_BOMEX_128x128x100_50CCN_50m",
        # "/wdata_visl/NEW_BOMEX/processed_HAWAII_2000CCN_512x220_50m",
        "/wdata_visl/NEW_BOMEX/processed_DYCOMS_RF02_512x160_50m_500CCN"
    ]
    for dataset in datasets:
        data_paths = [f for f in glob.glob(os.path.join(dataset, "*.mat"))]
        z_max = 0
        i=0
        for path in data_paths[50:]:
            print(i)
            i+=1
            x = sio.loadmat(path)
            lwc = x['lwc']
            z = x['z']
            lwc_z = np.sum(lwc, (0, 1))
            if np.sum(lwc_z)>0:
                z_max_curr = z[0,np.nonzero(lwc_z)[0][-1]]
                z_max = np.max([z_max,z_max_curr])

        print(z_max)



if __name__ == "__main__":
    cloud_tsne()
    # cloud_histograms()
