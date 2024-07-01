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
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

hydra.core.global_hydra.GlobalHydra.instance().clear()
CONFIG_DIR = os.path.join("../", "../", "configs")
N_cloud=200
override = False


# Load ProbCT model
def cloud_transform(x):
    x=np.log(np.abs(np.fft.fftn(x.cpu().numpy())))
    x = x.reshape(-1)
    if np.isfinite(x.sum(-1)):
        return x
    else:
        return None


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
    cfg.data.noise = False
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


def plot_pdfs(data, target_names, legend, bandwidth=5, grid_size=1):
    def construct_grids(data, bandwidth=5, grid_size=1):
        """Construct the map grid from the batch object

        Parameters
        ----------
        batch : Batch object
            The object returned by :func:`fetch_species_distributions`

        Returns
        -------
        (xgrid, ygrid) : 1-D arrays
            The grid corresponding to the values in batch.coverages
        """
        # x,y coordinates for corner cells
        xmin = data[:,1].min() - 2*bandwidth#+ batch.grid_size
        xmax = data[:,1].max() + 2*bandwidth
        ymin = data[:,0].min() - 2*bandwidth
        ymax = data[:,0].max() + 2*bandwidth

        # x coordinates of the grid cells
        xgrid = np.arange(xmin, xmax, grid_size)
        # y coordinates of the grid cells
        ygrid = np.arange(ymin, ymax, grid_size)

        return (xgrid, ygrid)

    # Set up the data grid for the contour plot
    xgrid, ygrid = construct_grids(data, bandwidth=bandwidth, grid_size=grid_size)
    X, Y = np.meshgrid(xgrid, ygrid)
    kde = KernelDensity(
        bandwidth=bandwidth,  kernel="gaussian"#, algorithm="ball_tree"
    )
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    # colors = 'r', 'g', 'b', 'c'
    colors = 'Reds', 'Greens', 'Blues', 'Purples'
    for c, label in zip(colors, legend):
        kde.fit(data[target_names == label])

        Z = np.exp(kde.score_samples(xy))
        Z = Z.reshape(X.shape)
        # Z[Z<Z.max()*0.1]=0
        # plot contours of the density
        levels = np.linspace(0, Z.max(), 100)
        color_array = getattr(plt.cm,c)(range(512))
        color_array[:,-1] = np.linspace(0.0,1.0,512)
        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name=f'{c}_alpha', colors=color_array)

        # register this new colormap with matplotlib
        plt.colormaps.register(cmap=map_object)

        plt.contourf(Y, X, Z, levels=levels, cmap=f'{c}_alpha')


def get_data(cfg, dataset_name):
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


def probct_inference(dataset_name):
    name = dataset_name.split('CCN')[0].replace('_','').replace('5000','500')
    MODEL_DIR = os.path.join("models", f"Test_{name}", f"Trained_{name}")
    model, cfg, device = load_model(MODEL_DIR=MODEL_DIR)
    model.to(device)
    pred_clouds = []
    val_dataloader = get_data(cfg, dataset_name=dataset_name)
    errors = []
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
                error = relative_error(ext_est=est_vols, ext_gt=gt_vol)
                if val_out["output_conf"] is not None:
                    conf_vol = conf_vol.squeeze().reshape(gt_vol.shape)
                    prob_vol = prob_vol.reshape(*gt_vol.shape, -1)
                else:
                    conf_vol = torch.empty(1)
                    prob_vol = torch.empty(1)
                if not np.isclose(grid[0][-1][1], 0.04):
                    est_vols = est_vols[:, :, ::2][..., :64]
                if grid[0][-1].shape[0] < 64:
                    s = list(est_vols.shape)
                    s[-1] = 64
                    z = torch.zeros(s, device=device)
                    z[..., :est_vols.shape[-1]] = est_vols
                    est_vols = z
                if est_vols.shape==(32,32,64):
                    est_vols=cloud_transform(est_vols)
                    if est_vols is not None:
                        pred_clouds.append(est_vols)
                        errors.append(error.cpu().detach().numpy())
        if len(pred_clouds)==N_cloud:
            break
    errors = np.array(errors)
    print(f'{name}: mean relative error {np.mean(errors)} with std of {np.std(errors)}')

    return pred_clouds


def cloud_tsne():
    output_path = 'outputs/cloud_data_for_tsne.pkl'
    output_path = os.path.abspath(output_path)

    datasets = [
        "/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/test",
        # '/wdata/roironen/Data/BOMEX_128x128x100_5000CCN_50m_micro_256/10cameras_20m/test',
        "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/test",
        "/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/test",
        "/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/test",
        # "/wdata/roironen/Data/DYCOMS_RF02_500CCN_64x64x159_50m/10cameras_20m/test",
        # "/wdata/roironen/Data/DYCOMS_RF02_50CCN_64x64x159_50m/10cameras_20m/test",

    ]
    exts = []
    grids = []
    legend = ["BOMEX50CCN", "BOMEX500CCN", "CASS600CCN", "HAWAII2000CCN", "DYCOMS_RF02_500CCN", "DYCOMS_RF02_50CCN"][
             :len(datasets)]
    names = []
    for dataset, name in zip(datasets, legend):
        data_paths = [f for f in glob.glob(os.path.join(dataset, "cloud*.pkl"))]
        ext = []
        grid = None
        for cloud_path in data_paths:
            with open(cloud_path, 'rb') as f:
                A = pickle.load(f)['ext']
                if len(ext) == 0 or A.shape == ext[-1].shape:
                    ext.append(A)
                if grid is None:
                    with open(cloud_path, 'rb') as f:
                        grid = pickle.load(f)['grid'][-1]
                        grids.append(grid)
        clouds = np.array(ext)
        if not np.isclose(grid[1] - grid[0], 0.04):
            clouds = clouds[:, :, :, ::2][..., :64]
        if grid.shape[-1] < 64:
            s = list(clouds.shape)
            s[-1] = 64
            z = np.zeros(s)
            z[...,:clouds.shape[-1]] = clouds
            clouds = z
        exts.append(np.log(np.abs(np.fft.fftn(clouds[:N_cloud], axes=(1, 2, 3)))))
        names += [name] * N_cloud
        X = np.vstack(exts)
        X = X.reshape(X.shape[0], -1)

    if not os.path.exists(output_path) or override:
        pred_clouds_bomex50 = probct_inference(dataset_name='BOMEX_50CCN_10cameras_20m')
        pred_clouds_bomex500 = probct_inference(dataset_name='BOMEX_5000CCN_new_10cameras_20m')
        pred_clouds_cass = probct_inference(dataset_name='CASS_600CCN_roiprocess_10cameras_20m')
        pred_clouds_hawaii = probct_inference(dataset_name='HAWAII_2000CCN_10cameras_20m')


        os.makedirs(''.join(output_path.split('/')[:-1]), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({'pred_clouds_bomex500':pred_clouds_bomex500,
                         'pred_clouds_cass':pred_clouds_cass, 'pred_clouds_bomex50':pred_clouds_bomex50, 'pred_clouds_hawaii':pred_clouds_hawaii}, f)


    else:
        with open(output_path, 'rb') as f:
            loaded_dict = pickle.load(f)
        pred_clouds_bomex500, pred_clouds_cass, pred_clouds_bomex50, pred_clouds_hawaii = loaded_dict.values()


    data = np.concatenate((X,pred_clouds_bomex50, pred_clouds_bomex500,pred_clouds_cass, pred_clouds_hawaii))

    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12)
    data = tsne.fit_transform(data)

    X_2d = data[:X.shape[0]]
    pred_2d_bomex50 = data[N_cloud * len(datasets):N_cloud * (len(datasets) + 1)]
    pred_2d_bomex500 = data[N_cloud * (len(datasets) + 1):N_cloud * (len(datasets) + 2)]
    pred_2d_cass = data[N_cloud * (len(datasets) + 2):N_cloud * (len(datasets) + 3)]
    pred_2d_hawaii = data[N_cloud * (len(datasets) + 3):N_cloud * (len(datasets) + 4)]
    fig, axs = plt.subplots(1, 1)

    colors = 'r', 'g', 'b', 'Purple'


    target_names = np.array(names)

    plot_pdfs(X_2d, target_names, legend, bandwidth=5, grid_size=1)
    # for c, label in zip(colors, legend):
    #     plt.scatter(X_2d[target_names == label, 0], X_2d[target_names == label, 1], c=c, label=label)

    plt.scatter(pred_2d_bomex50[:, 0], pred_2d_bomex50[:, 1], c=colors[0], label=legend[0], marker='x')
    plt.scatter(pred_2d_bomex500[:, 0], pred_2d_bomex500[:, 1], c=colors[1], label=legend[1], marker='x')
    plt.scatter(pred_2d_cass[:, 0], pred_2d_cass[:, 1], c=colors[2], label=legend[2], marker='x')
    plt.scatter(pred_2d_hawaii[:, 0], pred_2d_hawaii[:, 1], c=colors[3], label=legend[3], marker='x')
    fig.legend()

    fig.tight_layout()
    plt.axis('off')

    image_format = 'svg'  # e.g .png, .svg, etc.
    image_name = '/'.join(output_path.split('/')[:-1]+['cloud_tsne.svg'])

    fig.savefig(image_name, format=image_format, dpi=1200)
    plt.show()
    print()



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
    image_name = '../../outputs/data_hist.svg'

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
