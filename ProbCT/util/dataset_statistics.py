import pickle
import matplotlib.pyplot as plt
import glob, os
import scipy.io as sio
import numpy as np

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
    cloud_histograms()
