import copy
import numpy as np
import at3d
import shdom
import torch
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from collections import OrderedDict
import warnings
import pickle
from renderer.at3d_util import load_from_csv


def plot(gt_images,images,loss):
    plt.scatter(np.array(gt_images).ravel(),np.array(images).ravel())
    plt.axis('square')
    plt.show()
    f, axarr = plt.subplots(3, len(images))
    for ax, image,gt in zip(axarr.T, images,gt_images):
        ax[0].imshow(image)
        ax[1].imshow(gt)
        ax[2].imshow(np.abs(gt-image))
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
    plt.title(f'loss={loss}')
    plt.tight_layout()
    plt.show()


class LossAT3D(torch.autograd.Function):
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


class DiffRendererAT3D(object):
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
        self.scatterer_name = 'cloud'
        self.load_solver(cfg)


        self.state_gen = None
        self.n_jobs = cfg.shdom.n_jobs
        self.min_bound = cfg.cross_entropy.min
        self.max_bound = cfg.cross_entropy.max
        self.use_forward_grid = cfg.shdom.use_forward_grid
        with open('/wdata/roironen/Data/at3d_optical_property_generator.pkl', 'rb') as f:
            self.optical_property_generator = pickle.load(f)

        # L2 Loss
        self.image_mean = cfg.data.mean
        self.image_std = cfg.data.std
        self.loss_at3d = LossAT3D().apply
        self.loss_operator = torch.nn.Identity()


    def load_solver(self,cfg):
        # if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
        #     path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/solver2.pkl'
        # elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m' or cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
        #     path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/at3d.nc'

        if cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/at3d.nc'
            self.cloud_scatterer = load_from_csv(
                '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/cloud0.txt',
                density='lwc', origin=(0.0, 0.0))
        elif cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
            path = '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/at3d.nc'
            self.cloud_scatterer = load_from_csv(
                '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/cloud64785.txt',
                density='lwc', origin=(0.0, 0.0))
        elif cfg.data.dataset_name == 'BOMEX_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/at3d.nc'
            self.cloud_scatterer = load_from_csv(
                '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/cloud0.txt',
                density='lwc', origin=(0.0, 0.0))
        else:
            NotImplementedError()
        self.sensors, self.solvers, self.rte_grid = at3d.util.load_forward_model(path)
        self.wavelength = self.sensors.get_unique_solvers()[0]
        return

    def get_medium_estimator(self, cloud, mask):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground truth scatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """

        # Define the grid for reconstruction

        # optical_properties1 = self.solvers[self.wavelength].medium['cloud'].copy(deep=True)
        # optical_properties['veff'].values = np.full(optical_properties['veff'].shape, 0.1)
        # optical_properties['reff'].values = np.full(optical_properties['reff'].shape, 10)
        # optical_properties = optical_properties.drop_vars('extinction')
        cloud_scatterer = copy.deepcopy(self.cloud_scatterer)
        cloud_scatterer['veff'].values[:] = np.full(cloud_scatterer['veff'].shape, 0.1)
        # cloud_scatterer['reff'].values[np.isfinite(cloud_scatterer['reff'].values)] = 10
        cloud_scatterer['reff'].values[:] = np.full(cloud_scatterer['reff'].shape, 10)
        optical_properties = self.optical_property_generator(cloud_scatterer)[self.wavelength]
        ext = optical_properties['extinction'].copy(deep=True)
        ext.values = cloud
        optical_properties = optical_properties.drop_vars('extinction')

        grid_to_optical_properties = at3d.medium.GridToOpticalProperties(
            self.rte_grid, 'cloud', self.wavelength, optical_properties
        )

        # UnknownScatterers is a container for all of the unknown variables.
        # Each unknown_scatterer also records the transforms from the abstract state vector
        # to the gridded data in physical coordinates.
        unknown_scatterers = at3d.containers.UnknownScatterers(
            at3d.medium.UnknownScatterer(grid_to_optical_properties,
                                         extinction=(
                                             None, at3d.transforms.StateToGridMask(mask=mask)))
        )


        return unknown_scatterers


    def render(self, cloud, mask, gt_images):
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
        cloud[cloud<self.min_bound] = self.min_bound
        cloud[cloud>self.max_bound] = self.max_bound
        gt_images = gt_images.squeeze() # view x H x W
        gt_images *= self.image_std
        gt_images += self.image_mean
        gt_images = list(gt_images)

        unknown_scatterers = self.get_medium_estimator(mask.cpu().numpy())

        # now we form state_gen which updates the solvers with an input_state.
        solvers_reconstruct = at3d.containers.SolversDict()
        cloud_state = cloud[mask]
        cloud_state_np = cloud_state.detach().cpu().numpy()
        if self.state_gen is None:
            # prepare all of the static inputs to the solver just copy pasted from forward model
            surfaces = OrderedDict()
            numerical_parameters = OrderedDict()
            sources = OrderedDict()
            num_stokes = OrderedDict()
            background_optical_scatterers = OrderedDict()
            for key in self.sensors.get_unique_solvers():
                surfaces[key] = self.solvers[key].surface
                numerical_params = self.solvers[key].numerical_params
                numerical_params['num_mu_bins'] = 8
                numerical_params['num_phi_bins'] = 16
                numerical_parameters[key] = numerical_params
                sources[key] = self.solvers[key].source
                num_stokes[key] = self.solvers[key]._nstokes
                background_optical_scatterers[key] = {'rayleigh': self.solvers[key].medium['rayleigh']}
            # now we form state_gen which updates the solvers with an input_state.
            self.state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                                        unknown_scatterers, surfaces,
                                                        numerical_parameters, sources,
                                                        background_optical_scatterers,
                                                        num_stokes)

        else:
            self.state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                               unknown_scatterers, self.state_gen._surfaces,
                                               self.state_gen._numerical_parameters, self.state_gen._sources,
                                                    self.state_gen._background_optical_scatterers, self.state_gen._num_stokes)

        self.state_gen(cloud_state_np)

        for sensor, gt_image in zip(self.sensors['SideViewCamera']['sensor_list'],gt_images):
            sensor.I.data = gt_image.ravel('F')
        objective_function = at3d.optimize.ObjectiveFunction.LevisApproxUncorrelatedL2(
                    self.sensors, solvers_reconstruct, self.sensors, unknown_scatterers, self.state_gen,
                    self.state_gen.project_gradient_to_state,
                    parallel_solve_kwargs={'n_jobs': self.n_jobs, 'maxiter':100, 'verbose': False},
                    gradient_kwargs={'cost_function': 'L2', 'exact_single_scatter':True},
                    uncertainty_kwargs={'add_noise': False},
                    min_bounds=self.min_bound, max_bounds=self.max_bound)

        optimizer = at3d.optimize.Optimizer(objective_function)
        loss, gradient = optimizer.objective(cloud_state_np)

        gt_images = np.array(gt_images)
        images = self.sensors.get_images('SideViewCamera')
        images = [image.I.data for image in images]

        # plot(gt_images, images, loss)

        # print(np.sum(np.array(gt_images)-images)**2)
        self.loss = loss #/ np.sum(gt_images**2) #/ mask_images.sum()
        self.images = images
        self.gt_images = gt_images
        self.gradient = gradient
        l2_loss = self.loss_at3d(cloud_state,self) #/ gt_images.size #/ np.sum(gt_images**2)
        return self.loss_operator(l2_loss)


    def parallel_render(self, clouds, masks, gt_image_list):
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


        state_gen_list = []
        delayed_funcs = []
        for cloud, mask, gt_images in zip(clouds, masks, gt_image_list):
            cloud[0, :, :] = 0  # N x X x Y x Z
            cloud[-1, :, :] = 0
            cloud[:, 0, :] = 0
            cloud[:, -1, :] = 0
            cloud[:, :, -1] = 0
            cloud[cloud < self.min_bound] = self.min_bound
            cloud[cloud > self.max_bound] = self.max_bound
            unknown_scatterers = self.get_medium_estimator(cloud.detach().cpu().numpy(), mask.cpu().numpy())

            # now we form state_gen which updates the solvers with an input_state.
            solvers_reconstruct = at3d.containers.SolversDict()

            if self.state_gen is None:
                # prepare all of the static inputs to the solver just copy pasted from forward model
                surfaces = OrderedDict()
                numerical_parameters = OrderedDict()
                sources = OrderedDict()
                num_stokes = OrderedDict()
                background_optical_scatterers = OrderedDict()
                for key in self.sensors.get_unique_solvers():
                    surfaces[key] = self.solvers[key].surface
                    numerical_params = self.solvers[key].numerical_params
                    numerical_params['num_mu_bins'] = 8
                    numerical_params['num_phi_bins'] = 16
                    numerical_parameters[key] = numerical_params
                    sources[key] = self.solvers[key].source
                    num_stokes[key] = self.solvers[key]._nstokes
                    background_optical_scatterers[key] = {'rayleigh': self.solvers[key].medium['rayleigh']}
                # now we form state_gen which updates the solvers with an input_state.
                self.state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                                            unknown_scatterers, surfaces,
                                                            numerical_parameters, sources,
                                                            background_optical_scatterers,
                                                            num_stokes)

            # background_optical_scatterers[self.wavelength]['static_rayleigh': ]
            curr_state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                                   unknown_scatterers, self.state_gen._surfaces,
                                                   self.state_gen._numerical_parameters, self.state_gen._sources,
                                                        self.state_gen._background_optical_scatterers, self.state_gen._num_stokes)
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore")
            curr_sensors = copy.deepcopy(self.sensors)

            cloud_state = cloud[mask]
            cloud_state_np = cloud_state.detach().cpu().numpy()
            curr_state_gen(cloud_state_np)
            state_gen_list.append(cloud_state)

            gt_images = gt_images.squeeze()  # view x H x W
            gt_images *= self.image_std
            gt_images += self.image_mean
            gt_images = list(gt_images)
            for sensor, gt_image in zip(curr_sensors['SideViewCamera']['sensor_list'],gt_images):
                sensor.I.data = gt_image.ravel('F')
            objective_function = at3d.optimize.ObjectiveFunction.LevisApproxUncorrelatedL2(
                        curr_sensors, solvers_reconstruct, curr_sensors, unknown_scatterers, curr_state_gen,
                        curr_state_gen.project_gradient_to_state,
                        parallel_solve_kwargs={'n_jobs': self.n_jobs, 'maxiter':100, 'verbose': False},
                        gradient_kwargs={'cost_function': 'L2', 'exact_single_scatter':True},
                        uncertainty_kwargs={'add_noise': False},
                        min_bounds=self.min_bound, max_bounds=self.max_bound)

            optimizer = at3d.optimize.Optimizer(objective_function)
            # optimizer_list.append(optimizer)
            delayed_funcs.append(delayed(optimizer.objective)(cloud_state_np))
        parallel_pool = Parallel(n_jobs=self.n_jobs, backend='threading')
        out = parallel_pool(delayed_funcs)
        gradient = np.hstack([i[1] for i in out])
        loss = np.mean([i[0] for i in out])
        # loss, gradient = optimizer.objective(cloud_state.detach().cpu().numpy())
        state_gen_list = torch.hstack(state_gen_list)
        gt_images = np.array(gt_images)
        images = curr_sensors.get_images('SideViewCamera')
        images = np.array([image.I.data for image in images])

        # plot(gt_images, images, loss)

        # print(np.sum(np.array(gt_images)-images)**2)
        self.loss = loss #/ np.sum(gt_images**2) #/ mask_images.sum()
        self.images = images
        self.gt_images = gt_images
        self.gradient = gradient
        l2_loss = self.loss_at3d(state_gen_list,self) #/ gt_images.size #/ np.sum(gt_images**2)
        return self.loss_operator(l2_loss)
