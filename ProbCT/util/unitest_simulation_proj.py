# This file contains the a script for VIP-CT and ProbCT feature sampling unitest.
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

import pickle
import matplotlib.pyplot as plt
from scene.volumes import Volumes

import numpy as np
import torch
import torch.nn.functional as F
from scene.volumes import Volumes
from scene.cameras import PerspectiveCameras
from ProbCT.encoder.encoder import Backbone
# from scipy.interpolate import griddata
# from scipy import interpolate

if __name__ == "__main__":
    if True:
        # def sample_features(latents, uv):
        #     """
        #     Get pixel-aligned image features at 2D image coordinates
        #     :param latent (B, C, H, W) images features
        #     :param uv (B, N, 2) image points (x,y)
        #     :param image_size image size, either (width, height) or single int.
        #     if not specified, assumes coords are in [-1, 1]
        #     :return (B, C, N) L is latent size
        #     """
        #     uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        #     samples = torch.empty(0, device=uv.device)
        #     for latent in latents:
        #         samples = torch.cat((samples, torch.squeeze(F.grid_sample(
        #             latent,
        #             uv,
        #             align_corners=True,
        #             mode='bilinear',
        #             padding_mode='zeros',
        #         ))), dim=1)
        #     return samples  # (Cams,cum_channels, N)


        # with open('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/32cameras/train/cloud_results_0.pkl', 'rb') as outfile:
        # with open('/media/roironen/8AAE21F5AE21DB09/Data/CUBE.pkl', 'rb') as outfile:
        # with open('/wdata/roironen/Data/CASS_50m_256x256x139_600CCN/10cameras_20m/train/cloud_results_5555.pkl', 'rb') as outfile:
        with open('/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_50m/train/cloud_results_999.pkl', 'rb') as outfile:
            x = pickle.load(outfile)
        image_sizes = np.array([image.shape for image in x['images']])
        cameras = PerspectiveCameras(image_size=image_sizes[None], P=torch.tensor(x['cameras_P'],device='cuda').float(),
                                      device='cuda')
        # grid = torch.tensor(x['net_grid'],device='cuda')
        # grid = torch.tensor(x['grid'],device='cuda')

        extinction = x['ext'][None,None]
        mask = x['mask'][None]
        # mask = [torch.tensor(m) if mask is not None else m for m in mask]
        mask = [torch.ones(extinction.shape, device='cuda', dtype=bool)]
        layers = 4
        # images = [torch.arange(int(128/(i+1))**2).reshape(1,1,int(128/(i+1)),-1).double().repeat(image_sizes.shape[0],1,1,1) for i in range(layers)]
        images = x['images']
        grid2 = x['grid']
        # grid2[:-1] += 0.025
        # grid2[-1] += 0.02
        volume = Volumes(torch.tensor(extinction, device='cuda').float(), grid2)

        backbone = Backbone(backbone='resnet50_fpn',
                pretrained=False,
                num_layers=4,
                index_interp="bilinear",
                index_padding="border",
                upsample_interp="bilinear",
                feature_scale=1.0,
                use_first_pool=True,
                norm_type="batch",
                sampling_output_size=10,
                sampling_support = 10,
                out_channels = 1,
                n_sampling_nets=1,
                to_flatten = False,
                modify_first_layer=True).to('cuda')
        volume, query_points, _ = volume.get_query_points(2000, 'topk', masks=mask)
        uv = cameras.project_points(query_points, screen=True)
        # uv_swap= torch.zeros(uv[0].shape,device='cuda')
        # uv_swap[..., 0] = uv[0][..., 1]
        # uv_swap[..., 1] = uv[0][..., 0]
        # uv_swap = [uv_swap]

        boxes = [backbone.make_boxes(box_center).reshape(*box_center.shape[:2],-1) for box_center in uv]
        samples = backbone.sample_roi_debug([torch.tensor(images,device='cuda').unsqueeze(1).unsqueeze(1)],uv)
        # indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
        # print(torch.tensor(x['ext']).reshape(-1)[indices])
        # grid = x['grid']
        # volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
        samples[0] = samples[0].reshape(10,2000,10,10)
        N=2000
        # pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
        ind = np.random.permutation(2000)[:N]
        ind[0] = 0
        i=0
        for im, center in zip(images, uv[0]):
            fig, ax = plt.subplots(1)

            plt.imshow(im / np.max(im))
            plt.scatter(center[ind, 0].cpu().numpy(), center[ind, 1].cpu().numpy(), s=1, c='red',
                        marker='x')
            plt.title(i)
            i += 1
            plt.show()




        for im, box, center in zip(images, boxes[0], uv[0]):
            fig, ax = plt.subplots(1)
            colors = cm.YlOrBr(np.linspace(0, 1, N))

            plt.imshow(im / np.max(im))
            plt.scatter(center[ind, 0].cpu().numpy(), center[ind, 1].cpu().numpy(), s=1, c='red',
                        marker='x')
            # x = np.linspace(0, 115, 116)
            # y = np.linspace(0, 115, 116)
            # X, Y = np.meshgrid(x, y)
            # f = interpolate.interp2d(X, Y, im, kind='linear')
            for b, v, c in zip(box[ind], volume[0][ind],colors) :
                b = b.cpu().numpy()
                centerx = b[0]
                centery = b[1]
                h = (b[2] - b[0])
                w = (b[3] - b[1])

                # Ti = f(centery,centerx)
                # print(Ti)
                coord = np.round([centery,centerx]).astype(int)
                Ti = im[coord[0], coord[1]]
                Ti = round(Ti,3)
                # print(im[coord[1], coord[0]])
                # print(im[coord[0],coord[1]])
                # Ti = griddata((X, Y), im, np.array([centerx, centery]).reshape((1,2)), method='cubic')
                rect = patches.Rectangle((centerx, centery), w, h, linewidth=1,
                                         edgecolor=c, facecolor="none")
                plt.text(centerx+1, centery, f'{int(v)},{Ti:.3f}', fontsize=6,c =c)
                # Add the patch to the Axes
                ax.add_patch(rect)
            plt.title(i)
            i += 1
            plt.show()
        #     pdf.savefig(fig)
        # pdf.close()
        samples = samples[0][:,ind]
        # pdf = matplotlib.backends.backend_pdf.PdfPages("output1.pdf")

        fig, axs = plt.subplots(10,N)
        for im, sample, axx in zip(images, samples, axs):
            for i, ax in enumerate(axx):
                ax.imshow(sample.cpu().numpy()[i] / np.max(im))
                ax.axis('off')
        plt.show()
        # pdf.savefig(fig)
        # pdf.close()
        print()
    else:
        import os
        DEFAULT_DATA_ROOT = '/home/roironen/Data'
        data_root = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train'
        data_root = '/wdata/roironen/Data/CASS_50m_256x256x139_600CCN/10cameras_20m/train'
        if False:

            image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/AIRMSPI_IMAGES_LWC_LOW_SC/satellites_images_856.pkl'
            mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
            # mapping_path = '/wdata/yaelsc/AirMSPI_raw_data/raw_data/voxel_pixel_list72x72x32_BOMEX_img350x350.pkl'
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
            images_mapping_list = []
            for _, map in mapping.items():
                voxels_list = []
                v = map.values()
                voxels = np.array(list(v), dtype=object)
                ctr = 0
                for i, voxel in enumerate(voxels):
                    if len(voxel) > 0:
                        pixels = np.unravel_index(voxel, np.array([350, 350]))
                        mean_px = np.mean(pixels, 1)
                        voxels_list.append(mean_px)
                    else:
                        ctr += 1
                        voxels_list.append([-100000, -100000])
                images_mapping_list.append(voxels_list)

        else:
            import glob
            with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_images_mapping_lists32x32x32_BOMEX_img350x350.pkl',
                    'rb') as f:
                images_mapping_list = pickle.load(f)[1]
            with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_pixel_centers_lists32x32x32_BOMEX_img350x350.pkl',
                    'rb') as f:
                pixel_centers_lists = pickle.load(f)
            # image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/AIRMSPI_IMAGES_LWC_LOW_SC/satellites_images_856.pkl'
            image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/'
            image_root = [f for f in glob.glob(os.path.join(image_root, "SIMULATED_AIRMSPI_TRAIN*"))]
            image_root = [glob.glob(os.path.join(f, "*856.pkl")) for f in image_root][1][-1]

        # val_image = torch.tensor(val_images, device=device).float()[None]
        # masks = sio.loadmat('/wdata/yaelsc/AirMSPI_raw_data/raw_data/mask_72x72x32_vox50x50x40m.mat')['mask']

        image_index = image_root.split('satellites_images_')[-1].split('.pkl')[0]
        cloud_path = os.path.join(data_root, f"cloud_results_{image_index}.pkl")

        with open(image_root, 'rb') as f:
            images = pickle.load(f)['images']
        # images = sio.loadmat('/wdata/yaelsc/AirMSPI_raw_data/raw_data/croped_airmspi_9images_for_Roi.mat')[
        #     'croped_airmspi_images']
        image_sizes = np.array([[350, 350]] * 9)
        device = 'cuda'
        with open(cloud_path, 'rb') as f:
            x = pickle.load(f)
        masks = x['mask']*1.0
        masks = x['ext']
        EXT = torch.tensor(masks)
        indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
        N=int(np.sum(x['ext']>0))
        indices = torch.topk(EXT.reshape(-1), N).indices#[torch.randperm(N)[:1000]]


        mapping = [ np.array(map)[indices] for map in images_mapping_list]

        # cameras = AirMSPICameras(image_size=torch.tensor(image_sizes),
        #                          mapping=torch.tensor(mapping, device=device).float(),
        #                          device=device)

        layers = 4
        # print(torch.tensor(x['ext']).reshape(-1)[indices])
        grid = x['grid']
        # gx = np.linspace(0, 0.05 * 72, 72, dtype=np.float32)
        # gy = np.linspace(0, 0.05 * 72, 72, dtype=np.float32)
        # gz = np.linspace(0, 0.04 * 32, 32, dtype=np.float32)
        # grid = [np.array([gx, gy, gz])]
        volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
        # projected_to_screen = cameras.project_points(indices, screen=True)
        for im, screen_points in zip(images, mapping):
            mask = np.bitwise_and(screen_points[:, 0]>0, screen_points[:, 0]>0)
            plt.imshow(im / np.max(im))
            plt.scatter(screen_points[mask, 0], screen_points[mask, 1], s=1, c='red',
                        marker='x')
            plt.show()

        print()
