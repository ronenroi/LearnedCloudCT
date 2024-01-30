# This file contains auxiliry code for result visualization.
# It is based on pySHDOM source code ('https://github.com/aviadlevis/pyshdom') by Aviad Levis
# Copyright (c) Aviad Levis et al.
# All rights reserved.

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

import tensorboardX as tb
import matplotlib.pyplot as plt
import numpy as np

class SummaryWriter(object):
    """
    A wrapper for tensorboardX summarywriter with some basic summary writing implementation.
    This wrapper enables logging of images, error measures and loss with pre-determined temporal intervals into tensorboard.

    To view the summary of this run (and comparisons to all subdirectories):
        tensorboard --logdir LOGDIR

    Parameters
    ----------
    log_dir: str
        The directory where the log will be saved
    """
    def __init__(self, log_dir=None):
        self._dir = log_dir
        self._tf_writer = tb.SummaryWriter(log_dir) if log_dir is not None else None
        self._callback_fns = []
        self._kwargs = []
        self._iter = 0
        self._dataset = 'train'

    def add_callback_fn(self, callback_fn, kwargs=None):
        """
        Add a callback function to the callback function list

        Parameters
        ----------
        callback_fn: bound method
            A callback function to push into the list
        kwargs: dict, optional
            A dictionary with optional keyword arguments for the callback function
        """
        self._callback_fns.append(callback_fn)
        self._kwargs.append(kwargs)

    def monitor_loss(self, loss):
        """
        Monitor the loss.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'loss': loss,
            'ckpt_time': self._iter,
            'title': 'loss',
        }
        self.loss_cbfn(kwargs)

    def monitor_scatterer_error(self, delta, epsilon, name='extinction'):
        """

        """
        kwargs = {
            'delta': delta,
            'epsilon': epsilon,
            'title': [f'{name}_delta/', f'{name}_epsilon/'],
        }
        self.error_cbfn(kwargs)

    def monitor_scatter_plot(self, est_param, gt_param, ind=0, dilute_percent=0.8, name='extinction'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'est_param': est_param,
            'gt_param': gt_param,
            'title': '{}_scatter_plot/{}-{}'.format(name,self._dataset,ind),
            'percent': dilute_percent,
        }
        self.scatter_plot_cbfn(kwargs)

    def monitor_images(self, gt_images, est_images=[]):
        """
        Monitor the GT images

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        """
        num_images = gt_images.shape[0]

        vmax = [image.max() * 1.25 for image in gt_images]

        kwargs = {
            'images': gt_images,
            'title': ['Acquired/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax,
        }
        self.images_cbfn(kwargs)
        if len(est_images)>0:
            num_images = est_images.shape[0]
            kwargs = {
                'images': est_images,
                'title': ['Estimated/view{}'.format(view) for view in range(num_images)],
                'vmax': vmax,
            }
            self.images_cbfn(kwargs)


    def loss_cbfn(self,  kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalars(kwargs['title'], {
            self._dataset: kwargs['loss']}, self._iter)

    def images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self._iter, kwargs['images'], kwargs['title'], kwargs['vmax'])

    def error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """

        self.tf_writer.add_scalars(kwargs['title'][0], {
            self._dataset: kwargs['delta']}, self._iter)
        self.tf_writer.add_scalars(kwargs['title'][1], {
            self._dataset: kwargs['epsilon']}, self._iter)

    def scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        est_param = kwargs['est_param'].cpu().detach().numpy().ravel()
        gt_param = kwargs['gt_param'].cpu().detach().numpy().ravel()
        rho = np.corrcoef(est_param, gt_param)[1, 0]
        num_params = gt_param.size
        rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
        max_val = max(gt_param.max(), est_param.max())
        fig, ax = plt.subplots()
        ax.set_title(r' ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(100 * kwargs['percent'], rho),
                     fontsize=16)
        ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
        ax.set_xlim([0, 1.1*max_val])
        ax.set_ylim([0, 1.1*max_val])
        ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
        ax.set_ylabel('Estimated', fontsize=14)
        ax.set_xlabel('True', fontsize=14)

        self.tf_writer.add_figure(
            tag=kwargs['title'],
            figure=fig,
            global_step=self._iter
        )

    def write_image_list(self, global_step, images, titles, vmax=None):
        """
        Write an image list to tensorboardX.

        Parameters
        ----------
        global_step: integer,
            The global step of the optimizer.
        images: list
            List of images to be logged onto tensorboard.
        titles: list
            List of strings that will title the corresponding images on tensorboard.
        vmax: list or scalar, optional
            List or a single of scaling factor for the image contrast equalization
        """
        if np.isscalar(vmax) or vmax is None:
            vmax = [vmax]*len(images)

        assert len(images) == len(titles), 'len(images) != len(titles): {} != {}'.format(len(images), len(titles))
        assert len(vmax) == len(titles), 'len(vmax) != len(images): {} != {}'.format(len(vmax), len(titles))

        for image, title, vm in zip(images, titles, vmax):
            if (image.shape[0] in (1,3,4)): #polarized
                #there is some overlap in this condition with a multispectral unpolarized case
                # with a very small number of pixels in the first dimension.
                image = image[0]

            if image.ndim==3: # polychromatic
                img_tensor = image[:,:,:]/ image.max()

            else:
                img_tensor = image[:,:,np.newaxis] / image.max()
            self.tf_writer.add_images(tag=title,
                img_tensor=img_tensor,
                dataformats='HWN',
                global_step=global_step
                )

    @property
    def callback_fns(self):
        return self._callback_fns

    @property
    def dir(self):
        return self._dir

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def tf_writer(self):
        return self._tf_writer

