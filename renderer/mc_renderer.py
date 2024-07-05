import os, time
import numpy as np
import argparse
import shdom
import dill as pickle
import torch
import matplotlib.pyplot as plt
from MC_renderer.cuda_utils import float_reg
from renderer.shdom_util import Monotonous
from MC_renderer.classes.scene_seed_roi import *
from MC_renderer.classes.camera import Camera2 as MCcamera
from MC_renderer.classes.camera import Camera as CameraO
from MC_renderer.classes.volume import Volume as MCvolume
from MC_renderer.classes.grid import Grid as MCgrid
class LossSHDOM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, optimizer):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # gradient = np.zeros(input.shape)
        # gradient[optimizer.mask.data] = optimizer.gradient
        gradient = torch.tensor(optimizer.gradient, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(gradient)
        return torch.tensor(optimizer.loss, dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        gradient, = ctx.saved_tensors
        return gradient * grad_output.clone(), None


class LossMC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, optimizer):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        print("RESAMPLING PATHS ")

        # I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=measurements.detach().cpu().numpy(), to_torch=True)
        I_opt, total_grad = optimizer.render(I_gt=self.measurements, to_torch=True)
        total_grad *= (measurements.numel())
        I_opt = torch.tensor(I_opt, dtype=input.dtype,device=total_grad.device) #* (measurements.numel())
        error = torch.nn.MSELoss()
        scene_rr.I_opt = I_opt.detach().cpu().numpy()
        print('Done image consistency loss stage')
        total_grad[torch.logical_not(torch.isfinite(total_grad))] = 0
        ctx.save_for_backward(torch.squeeze(total_grad))
        loss = error(I_opt,measurements) * (measurements.numel())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # import pdb
        # pdb.set_trace()
        gradient, = ctx.saved_tensors
        # print(gradient.shape)
        return gradient * grad_output.clone(), None, None, None


class DiffRendererMC(object):
    """
    Optimize: Extinction
    --------------------
    Estimate the extinction coefficient based on monochrome radiance measurements.
    In this script, the phase function, albedo and rayleigh scattering are assumed known and are not estimated.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_extinction_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.scatterer_name = 'cloud'
        self.cameras = self.get_cameras(cfg)
        self.get_sun()
        self.min_bound = cfg.cross_entropy.min
        self.max_bound = cfg.cross_entropy.max
        self.add_rayleigh = cfg.renderer.add_rayleigh
        self.Npath = int(1e8)
        self.Npath_times = 1
        self.im_background = 0.0176

        self.use_forward_grid = cfg.renderer.use_forward_grid
        # self.device = device
        # parser = argparse.ArgumentParser()
        # CloudGenerator = getattr(shdom.generate, 'Homogenous')
        # CloudGenerator = Monotonous
        # parser = CloudGenerator.update_parser(parser)
        #


        # self.args = parser.parse_args()
        # self.args.air_max_alt = 5
        # self.args.extinction = 0



        # L2 Loss
        self.image_mean = cfg.data.mean
        self.image_std = cfg.data.std
        self.loss_shdom = LossSHDOM().apply
        self.loss_operator = torch.nn.Identity()

    def get_grid(self, grid):
        nx = grid[0].shape[0]
        ny = grid[1].shape[0]
        nz = grid[2].shape[0]
        voxel_size_x = grid[0][1] - grid[0][0]
        voxel_size_y = grid[1][1] - grid[1][0]
        voxel_size_z = grid[2][1] - grid[2][0]
        edge_x = voxel_size_x * nx
        edge_y = voxel_size_y * ny
        edge_z = voxel_size_z * nz
        bbox = np.array([[0, edge_x],
                         [0, edge_y],
                         [0, edge_z]])
        grid = MCgrid(bbox, (nx,ny,nz))
        return grid

    def get_sun(self):
        if self.cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
            path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/solver2.pkl'
        elif self.cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m' or 'BOMEX_50CCN_aux_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/solver.pkl'
        else:
            NotImplementedError()
        solver = shdom.RteSolver()
        solver.load_params(path)
        # params = solver._numerical_parameters
        # params.num_mu_bins = 2
        # params.num_phi_bins = 4
        # solver.set_numerics(params)
        sun_azimuth = solver._scene_parameters.source.azimuth
        sun_zenith = solver._scene_parameters.source.zenith
        self.sun_angles = np.array([sun_zenith, sun_azimuth]) * (np.pi / 180)

        

    def get_cameras(self,cfg):
        if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
            path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/shdom_projections2.pkl'
        elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m' or cfg.data.dataset_name == 'BOMEX_50CCN_aux_10cameras_20m' or cfg.data.dataset_name == 'BOMEX_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/shdom_projections.pkl'
        elif cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
            path = '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/shdom_projections.pkl'
            NotImplementedError()
        with open(path, 'rb') as pickle_file:
            projection_list = pickle.load(pickle_file)['projections']
        cameras = []
        volume_center = np.array([1.6/2,1.6/2,0.68 * 32 * 0.04]) #/ 2
        t_list=[]
        phi_list=[]
        R_list = []
        for projection in projection_list:
            # euler_angles = np.array((180 - projection.azimuth, 0, projection.zenith - 90))
            # T_world = projection.position
            # T = -projection._rotation_matrix.T @ T_world
            from scipy.spatial.transform import Rotation

            r = Rotation.from_matrix(projection._rotation_matrix)
            angles = r.as_euler("xyz", degrees=True)
            n_angles = np.array([angles[0],angles[2],angles[1]])
            phi_list.append(angles[1])
            t_list.append(projection.position)
            R_list.append(projection._rotation_matrix)
            # print(n_angles)
            r = Rotation.from_euler("xyz",n_angles,degrees=True).as_matrix()
            t = projection.position + volume_center
            # t_new = np.array(())
            # camera = CameraO(t=projection.position + volume_center, euler_angles=n_angles, focal_length=projection._focal, sensor_size=np.array((2, 2)),pixels=projection.resolution)
            # print(camera.t)
            # camera = MCcamera(t=projection.position + volume_center, R=r, focal_length=projection._focal, sensor_size=np.array((2, 2)),pixels=projection.resolution)
            # cameras.append(camera)
        volume_center = np.array([1.6/2, 1.6/2, 0]) # ???

        N_cams = 10
        cameras1 = []
        # volume_center[-1] *= 1.8
        # edge_z = bbox[-1, -1]
        # R = 500
        focal_length = projection._focal #1e-4  #####################
        sensor_size = np.array((2, 2))#np.array((3e-4, 3e-4)) / 2  #####################
        ps_max = 116
        pixels = np.array((ps_max, ps_max))

        cam_deg = 360 // (N_cams - 1)
        for cam_ind in range(N_cams - 1 +1):
            theta = 0
            theta_rad = theta * (np.pi / 180)
            # phi = (-(N_cams // 2) + cam_ind) * cam_deg
            # print((-(N_cams // 2) + cam_ind) * cam_deg)
            phi = phi_list[cam_ind]#-180
            # print(phi)
            phi_rad = phi * (np.pi / 180)

            t = t_list[cam_ind] #+ volume_center #R * theta_phi_to_direction(theta_rad, phi_rad) #+ volume_center

            euler_angles = np.array((180 - theta, phi, 180))
            # print(euler_angles)
            # camera = CameraO(t, euler_angles, focal_length, sensor_size, pixels)
            camera = MCcamera(t, R_list[cam_ind], focal_length, sensor_size, pixels)

            # print(camera.t)

            cameras1.append(camera)
        # print(phi_list)
        # t = R * theta_phi_to_direction(0, 0) + volume_center
        # euler_angles = np.array((180, 0, -90))
        # cameras1.append(CameraO(t, euler_angles, cameras1[0].focal_length, cameras1[0].sensor_size, cameras1[0].pixels))
        return cameras1


    def get_medium_estimator(self, cloud_extinction, mask, volume):
        """
        Generate the medium estimator for optimization.

        """

        # Define the grid for reconstruction
        if self.use_forward_grid:
            grid = self.get_grid(volume._grid[0])

        else:
            NotImplementedError()
            # extinction_grid = albedo_grid = phase_grid = self.get_grid()


        # Define the known albedo and phase: either ground-truth or specified, but it is not optimized.


        if self.add_rayleigh:
            beta_air = 0.004
        else:
            beta_air = 0.0
        w0_air = 0.912
        w0_cloud = 0.99
        medium_estimator = MCvolume(grid, cloud_extinction, beta_air, w0_cloud, w0_air)
        medium_estimator.set_mask(mask)

        return medium_estimator


    def init_optimizer(self):
        """
        Initialize the optimizer.
        This means:
          1. Setting the RteSolver medium
          2. Initializing a solution
          3. Computing the direct solar flux derivatives
          4. Counting the number of unknown parameters
        """

        assert self.measurements.num_channels == self.medium.num_wavelengths, \
            'Measurements have {} channels and Medium has {} wavelengths'.format(
                self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = self.medium.num_parameters

    def create_renderer(self, volume):

        ########################
        # Atmosphere parameters#
        ########################

        g_cloud = 0.85
        rr_depth = 20
        rr_stop_prob = 0.05

        scene_rr = PytorchSceneSeed(volume, self.cameras, self.sun_angles, g_cloud, rr_depth, rr_stop_prob,
                                         N_batches=self.Npath_times,
                                         device=self.device)
        # scene_rr.upscale_cameras(ps_max)
        scene_rr.init_cuda_param(self.Npath, init=True)
        scene_rr.build_paths_list(self.Npath)
        scene_rr.set_cloud_mask(volume.cloud_mask)

        return scene_rr

    def shdom2mc_transform(self, shdom_images):
        mc_grid = self.medium.grid
        mc_images = shdom_images.astype(np.float32)
        mc_images[mc_images<=self.im_background*1.05] = 0
        ratio = (mc_grid.bbox_size[0] * mc_grid.bbox_size[1] * 1e6)  * 1e3 #/ (np.cos(self.sun_angles[1]))
        mc_images /= ratio

        return mc_images

    def mc2shdom_transform(self, mc_images):
        mc_grid = self.medium.grid
        shdom_images = mc_images
        ratio = (mc_grid.bbox_size[0] * mc_grid.bbox_size[1] * 1e6)  * 1e3 #/ (np.cos(self.sun_angles[1]))
        shdom_images *= ratio
        shdom_images[shdom_images <= self.im_background/2] = self.im_background
        return shdom_images
    
    def render(self, cloud, mask, volume, gt_images):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        cloud[0,:,:] = 0
        cloud[-1,:,:] = 0
        cloud[:, 0, :] = 0
        cloud[:, -1, :] = 0
        cloud[:, :,-1] = 0
        self.device = cloud.device
        gt_images = gt_images.squeeze()
        gt_images *= self.image_std
        gt_images += self.image_mean


        self.medium = self.get_medium_estimator(cloud.detach().cpu().numpy(), mask.cpu().numpy(), volume)
        mc_renderer = self.create_renderer(self.medium)

        images, gradient = mc_renderer.render(I_gt=self.shdom2mc_transform(gt_images), to_torch=True)
        gradient *= (gt_images.shape[-2]*gt_images.shape[-1])#(gt_images.size)
        images = self.mc2shdom_transform(images) #(gt_images.shape[-2]*gt_images.shape[-1])
        # I_opt = torch.tensor(I_opt, dtype=input.dtype, device=total_grad.device)  # * (measurements.numel())
        # scene_rr.I_opt = I_opt.detach().cpu().numpy()
        print('Done image consistency loss stage')
        gradient[torch.logical_not(torch.isfinite(gradient))] = 0
        # cloud_estimator = self.medium.scatterers['cloud']
        # cloud_mask = shdom.GridData(cloud_estimator.grid, (mask.cpu().numpy()))
        # cloud_estimator.set_mask(cloud_mask)


        # vmax = max(gt_images[0,5].max().item(),images[5].max())
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(gt_images[0,4])#,vmin=0,vmax=vmax
        # axarr[1].imshow(images[4])
        # # axarr[2].imshow(np.abs(gt_images[0,5] - images[5]))
        # plt.show()
        #
        # plt.scatter(gt_images.ravel(),np.array(images).ravel())
        # plt.axis('square')
        # plt.show()
        # f, axarr = plt.subplots(2, images.shape[0], figsize=(16, 16))
        # for ax, image in zip(axarr[0], images):
        #     ax.imshow(image)
        #     ax.invert_xaxis()
        #     ax.invert_yaxis()
        #     ax.axis('off')
        # for ax, image in zip(axarr[1], gt_images):
        #     ax.imshow(image)
        #     ax.invert_xaxis()
        #     ax.invert_yaxis()
        #     ax.axis('off')
        # plt.show()
        # self.loss = np.sum((images - gt_images)**2)
        self.images = images
        self.gt_images = gt_images
        # self.gradient = gradient
        # l2_loss = self.loss_mc(cloud_state,self) / np.sum(gt_images**2) #gt_images.size
        # return self.loss_operator(l2_loss)
        return None




