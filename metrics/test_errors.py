# This file contains the test error criteria for VIP-CT evaluation.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite our papers described in the readme file.:
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
import torch


# error criteria
def relative_error(ext_est, ext_gt, eps=1e-6):
    assert len(ext_est.shape)<=3
    assert len(ext_gt.shape)<=3
    rel_error = torch.norm(ext_est.view(-1) - ext_gt.view(-1), p=1) / (torch.norm(ext_gt.view(-1), p=1) + eps)
    return rel_error

def relative_squared_error(ext_est, ext_gt, eps=1e-6):
    assert len(ext_est.shape)<=3
    assert len(ext_gt.shape)<=3
    rel_squared_error = torch.sum((ext_est.view(-1) - ext_gt.view(-1)) ** 2) / (torch.sum((ext_gt.view(-1)) ** 2) + eps)
    return rel_squared_error

def relative_mass_error(ext_est, ext_gt, eps=1e-6):
    assert len(ext_est.shape)<=3
    assert len(ext_gt.shape)<=3
    rel_mass_error = (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)
    return rel_mass_error

def abs_error(ext_est, ext_gt):
    assert len(ext_est.shape)<=3
    assert len(ext_gt.shape)<=3
    error = torch.mean(torch.abs(ext_est.view(-1) - ext_gt.view(-1)))
    return error

def relative_voxel_error(ext_est, ext_gt, eps=1):
    assert len(ext_est.shape)<=3
    assert len(ext_gt.shape)<=3
    rel_error = torch.mean(torch.abs(ext_est.view(-1) - ext_gt.view(-1)) / (ext_gt.view(-1) + eps))
    return rel_error
