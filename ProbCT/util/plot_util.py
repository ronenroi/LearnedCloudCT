# This file contains util routines for VIP-CT.
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

import numpy as np
import matplotlib.pyplot as plt


def show_scatter_plot(gt_param, est_param):
    gt_param = gt_param.detach().cpu().numpy().ravel()
    est_param = est_param.detach().cpu().numpy().ravel()
    max_val = max(gt_param.max(), est_param.max())
    fig, ax = plt.subplots()
    ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
    ax.set_xlim([0, 1.1 * max_val])
    ax.set_ylim([0, 1.1 * max_val])
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')
    plt.show()

def show_scatter_plot_altitute(gt_param, est_param):
    import matplotlib.cm as cm
    gt_param = gt_param.detach().cpu().numpy()
    est_param = est_param.detach().cpu().numpy()
    colors = cm.rainbow(np.linspace(0, 1, gt_param.shape[-1]))

    max_val = max(gt_param.max(), est_param.max())
    fig, ax = plt.subplots()
    for i, c in enumerate(colors):
        if i>10:
            ax.scatter(gt_param[...,i].ravel(), est_param[...,i].ravel(),
                   facecolors='none', edgecolors=c, label=i)
        else:
            ax.scatter(gt_param[..., i].ravel(), est_param[..., i].ravel(),
                       facecolors='none', edgecolors=c)
    ax.set_xlim([0, 1.1 * max_val])
    ax.set_ylim([0, 1.1 * max_val])
    ax.legend(loc='best',fontsize='small')
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')
    plt.show()

def volume_plot(gt_param, est_param):
    gt_param = gt_param.detach().cpu().numpy()
    est_param = est_param.detach().cpu().numpy()
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("GT")
    ax.voxels(gt_param)
    plt.show()
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("Est.")
    ax.voxels(est_param)
    plt.show()


